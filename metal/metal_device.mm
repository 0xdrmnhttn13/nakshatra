#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_backend.hpp"

#include "tensor.hpp"

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace nakshatra {

namespace {

const char *kKernelSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void matvec_naive(
    device const float* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y       [[buffer(2)]],
    constant uint& in_dim [[buffer(3)]],
    uint row              [[thread_position_in_grid]])
{
    float sum = 0.0f;
    const uint base = row * in_dim;
    for (uint i = 0; i < in_dim; ++i) {
        sum += W[base + i] * x[i];
    }
    y[row] = sum;
}

kernel void matvec_naive_bf16(
    device const ushort* Wb [[buffer(0)]],
    device const float* x   [[buffer(1)]],
    device float* y         [[buffer(2)]],
    constant uint& in_dim   [[buffer(3)]],
    uint row                [[thread_position_in_grid]])
{
    float sum = 0.0f;
    const uint base = row * in_dim;
    for (uint i = 0; i < in_dim; ++i) {
        const float w = float(as_type<half>(Wb[base + i]));
        sum += w * x[i];
    }
    y[row] = sum;
}
)METAL";

} // namespace

struct MetalMatvec::Impl {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> pipeline = nil;
  id<MTLComputePipelineState> pipeline_bf16 = nil;

  WeightFormat format = WeightFormat::BF16;

  id<MTLBuffer> x_buf = nil;
  std::size_t x_capacity = 0;
  id<MTLBuffer> y_buf = nil;
  std::size_t y_capacity = 0;

  struct WeightEntry {
    id<MTLBuffer> buffer;
    bool bf16;
  };

  std::unordered_map<const void *, WeightEntry> weight_cache;

  WeightEntry weight_buffer(const float *w, std::size_t out,
                            std::size_t in) {
    auto it = weight_cache.find(w);
    if (it != weight_cache.end()) {
      return it->second;
    }

    WeightEntry entry;
    entry.bf16 = format == WeightFormat::BF16;

    if (entry.bf16) {
      const std::size_t n = out * in;
      std::vector<std::uint16_t> bf16(n);
      for (std::size_t i = 0; i < n; ++i) {
        bf16[i] = float_to_bf16_trunc(w[i]);
      }
      entry.buffer = [device
          newBufferWithBytes:bf16.data()
                      length:n * sizeof(std::uint16_t)
                     options:MTLResourceStorageModeShared];
    } else {
      entry.buffer = [device
          newBufferWithBytes:w
                      length:out * in * sizeof(float)
                     options:MTLResourceStorageModeShared];
    }

    weight_cache.emplace(w, entry);
    return entry;
  }

  bool ensure_x(std::size_t in) {
    if (x_capacity >= in * sizeof(float)) {
      return true;
    }
    x_buf = [device newBufferWithLength:in * sizeof(float)
                                options:MTLResourceStorageModeShared];
    x_capacity = in * sizeof(float);
    return x_buf != nil;
  }

  bool ensure_y(std::size_t out) {
    if (y_capacity >= out * sizeof(float)) {
      return true;
    }
    y_buf = [device newBufferWithLength:out * sizeof(float)
                                options:MTLResourceStorageModeShared];
    y_capacity = out * sizeof(float);
    return y_buf != nil;
  }
};

MetalMatvec::MetalMatvec() : impl_(std::make_unique<Impl>()) {
  @autoreleasepool {
    impl_->device = MTLCreateSystemDefaultDevice();
    if (impl_->device == nil) {
      return;
    }

    impl_->queue = [impl_->device newCommandQueue];
    if (impl_->queue == nil) {
      return;
    }

    NSError *error = nil;
    id<MTLLibrary> library = [impl_->device
        newLibraryWithSource:[NSString stringWithUTF8String:kKernelSource]
                      options:nil
                        error:&error];
    if (library == nil) {
      return;
    }

    id<MTLFunction> function = [library newFunctionWithName:@"matvec_naive"];
    if (function == nil) {
      return;
    }

    impl_->pipeline = [impl_->device
        newComputePipelineStateWithFunction:function
                                       error:&error];

    id<MTLFunction> function_bf16 =
        [library newFunctionWithName:@"matvec_naive_bf16"];
    if (function_bf16 == nil) {
      return;
    }

    impl_->pipeline_bf16 = [impl_->device
        newComputePipelineStateWithFunction:function_bf16
                                       error:&error];
  }
}

MetalMatvec::~MetalMatvec() = default;

MetalMatvec &global_metal() {
  static MetalMatvec instance;
  return instance;
}

bool MetalMatvec::ok() const {
  return impl_ != nullptr && impl_->pipeline != nil;
}

bool MetalMatvec::enabled() const { return enabled_; }

void MetalMatvec::set_enabled(bool on) { enabled_ = on; }

void MetalMatvec::set_weight_format(WeightFormat fmt) {
  if (impl_->format != fmt) {
    impl_->weight_cache.clear();
    impl_->format = fmt;
  }
}

WeightFormat MetalMatvec::weight_format() const { return impl_->format; }

bool MetalMatvec::run(const float *w, const float *x, float *y,
                      std::size_t out, std::size_t in) {
  if (!ok()) {
    return false;
  }

  @autoreleasepool {
    const NSUInteger w_bytes = out * in * sizeof(float);
    const NSUInteger x_bytes = in * sizeof(float);
    const NSUInteger y_bytes = out * sizeof(float);

    id<MTLBuffer> w_buf = [impl_->device
        newBufferWithBytes:w
                    length:w_bytes
                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> x_buf = [impl_->device
        newBufferWithBytes:x
                    length:x_bytes
                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> y_buf = [impl_->device
        newBufferWithLength:y_bytes
                   options:MTLResourceStorageModeShared];

    if (w_buf == nil || x_buf == nil || y_buf == nil) {
      return false;
    }

    uint32_t in_dim = static_cast<uint32_t>(in);

    id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:impl_->pipeline];
    [enc setBuffer:w_buf offset:0 atIndex:0];
    [enc setBuffer:x_buf offset:0 atIndex:1];
    [enc setBuffer:y_buf offset:0 atIndex:2];
    [enc setBytes:&in_dim length:sizeof(in_dim) atIndex:3];

    const MTLSize grid = MTLSizeMake(out, 1, 1);
    const MTLSize group = MTLSizeMake(64, 1, 1);

    [enc dispatchThreads:grid threadsPerThreadgroup:group];
    [enc endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(y, [y_buf contents], y_bytes);
  }

  return true;
}

bool MetalMatvec::matvec(const float *w, const float *x, float *y,
                         std::size_t out, std::size_t in) {
  if (!ok() || !enabled_) {
    return false;
  }

  @autoreleasepool {
    const Impl::WeightEntry w_entry = impl_->weight_buffer(w, out, in);
    if (w_entry.buffer == nil || !impl_->ensure_x(in) ||
        !impl_->ensure_y(out)) {
      return false;
    }

    memcpy([impl_->x_buf contents], x, in * sizeof(float));

    uint32_t in_dim = static_cast<uint32_t>(in);

    id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:w_entry.bf16 ? impl_->pipeline_bf16
                                            : impl_->pipeline];
    [enc setBuffer:w_entry.buffer offset:0 atIndex:0];
    [enc setBuffer:impl_->x_buf offset:0 atIndex:1];
    [enc setBuffer:impl_->y_buf offset:0 atIndex:2];
    [enc setBytes:&in_dim length:sizeof(in_dim) atIndex:3];

    const MTLSize grid = MTLSizeMake(out, 1, 1);
    const MTLSize group = MTLSizeMake(64, 1, 1);

    [enc dispatchThreads:grid threadsPerThreadgroup:group];
    [enc endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(y, [impl_->y_buf contents], out * sizeof(float));
  }

  return true;
}

bool MetalMatvec::matvec_batch(const MetalJob *jobs, std::size_t count) {
  if (!ok() || !enabled_ || count == 0) {
    return false;
  }

  @autoreleasepool {
    std::size_t x_total = 0;
    std::size_t y_total = 0;
    for (std::size_t j = 0; j < count; ++j) {
      x_total += jobs[j].in;
      y_total += jobs[j].out;
    }

    if (!impl_->ensure_x(x_total + 16) || !impl_->ensure_y(y_total + 16)) {
      return false;
    }

    std::vector<std::size_t> x_off(count);
    std::vector<std::size_t> y_off(count);

    std::size_t x_cursor = 0;
    for (std::size_t j = 0; j < count; ++j) {
      x_off[j] = x_cursor;
      memcpy(static_cast<std::uint8_t *>([impl_->x_buf contents]) + x_cursor,
             jobs[j].x, jobs[j].in * sizeof(float));
      x_cursor += (jobs[j].in * sizeof(float) + 15) & ~std::size_t(15);
    }

    id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    std::size_t y_cursor = 0;
    for (std::size_t j = 0; j < count; ++j) {
      const Impl::WeightEntry we =
          impl_->weight_buffer(jobs[j].w, jobs[j].out, jobs[j].in);
      if (we.buffer == nil) {
        return false;
      }

      uint32_t in_dim = static_cast<uint32_t>(jobs[j].in);

      [enc setComputePipelineState:we.bf16 ? impl_->pipeline_bf16
                                          : impl_->pipeline];
      [enc setBuffer:we.buffer offset:0 atIndex:0];
      [enc setBuffer:impl_->x_buf offset:x_off[j] atIndex:1];
      [enc setBuffer:impl_->y_buf offset:y_cursor atIndex:2];
      [enc setBytes:&in_dim length:sizeof(in_dim) atIndex:3];
      [enc dispatchThreads:MTLSizeMake(jobs[j].out, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

      y_off[j] = y_cursor;
      y_cursor += (jobs[j].out * sizeof(float) + 15) & ~std::size_t(15);
    }

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    for (std::size_t j = 0; j < count; ++j) {
      memcpy(jobs[j].y,
             static_cast<const std::uint8_t *>([impl_->y_buf contents]) +
                 y_off[j],
             jobs[j].out * sizeof(float));
    }
  }

  return true;
}

double MetalMatvec::bench(const float *w, const float *x, float *y,
                          std::size_t out, std::size_t in,
                          std::size_t warmup, std::size_t iters) {
  if (!ok()) {
    return -1.0;
  }

  @autoreleasepool {
    const NSUInteger w_bytes = out * in * sizeof(float);
    const NSUInteger x_bytes = in * sizeof(float);
    const NSUInteger y_bytes = out * sizeof(float);

    id<MTLBuffer> w_buf = [impl_->device
        newBufferWithBytes:w
                    length:w_bytes
                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> x_buf = [impl_->device
        newBufferWithBytes:x
                    length:x_bytes
                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> y_buf = [impl_->device
        newBufferWithLength:y_bytes
                   options:MTLResourceStorageModeShared];

    uint32_t in_dim = static_cast<uint32_t>(in);

    const MTLSize grid = MTLSizeMake(out, 1, 1);
    const MTLSize group = MTLSizeMake(64, 1, 1);

    auto dispatch = [&]() {
      id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      [enc setComputePipelineState:impl_->pipeline];
      [enc setBuffer:w_buf offset:0 atIndex:0];
      [enc setBuffer:x_buf offset:0 atIndex:1];
      [enc setBuffer:y_buf offset:0 atIndex:2];
      [enc setBytes:&in_dim length:sizeof(in_dim) atIndex:3];
      [enc dispatchThreads:grid threadsPerThreadgroup:group];
      [enc endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
    };

    for (std::size_t i = 0; i < warmup; ++i) {
      dispatch();
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iters; ++i) {
      dispatch();
    }
    const auto t1 = std::chrono::steady_clock::now();

    memcpy(y, [y_buf contents], y_bytes);

    return std::chrono::duration<double>(t1 - t0).count() /
           static_cast<double>(iters);
  }
}

} // namespace nakshatra

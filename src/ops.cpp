#include "ops.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

#include <span>

namespace nakshatra {
float dot(const float *a, const float *b, std::size_t n) {
  float sum = 0.0f;
  for (std::size_t i = 0; i < n; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

void linear(std::span<const float> x, std::span<const float> weight,
            std::size_t out_features, std::size_t in_features,
            std::span<float> y) {
  for (std::size_t o = 0; o < out_features; ++o) {
    y[o] = dot(x.data(), weight.data() + o * in_features, in_features);
  }
}

void gemma_rmsnorm(std::span<const float> x, std::span<const float> weight,
                   float eps, std::span<float> y) {
  float mean_sq = 0.0f;
  for (float v : x) {
    mean_sq += v * v;
  }
  mean_sq /= static_cast<float>(x.size());
  const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);
  for (std::size_t i = 0; i < x.size(); ++i) {
    y[i] = x[i] * inv_rms * (1.0f + weight[i]);
  }
}

float gelu_tanh(float x) {
  constexpr float k = 0.7978845608028654f;
  constexpr float c = 0.044715f;
  const float inner = k * (x + c * x * x * x);
  return 0.5f * x * (1.0f + std::tanh(inner));
}

void gemma_mlp(std::span<const float> x, std::span<const float> gate_w,
               std::span<const float> up_w, std::span<const float> down_w,
               std::size_t hidden, std::size_t intermediate,
                std::span<float> y) {

  std::vector<float> gate(intermediate);
  std::vector<float> up(intermediate);
  std::vector<float> fused(intermediate);

  linear(x, gate_w, intermediate, hidden, gate);
  linear(x, up_w, intermediate, hidden, up);

  for (std::size_t i = 0; i < intermediate; ++i) {
    fused[i] = gelu_tanh(gate[i]) * up[i];
  }

  linear(fused, down_w, hidden, intermediate, y);
}

void softmax_inplace(std::span<float> x) {
  const float max_v = *std::max_element(x.begin(), x.end());

  float sum = 0.0f;

  for (float &v : x) {
    v = std::exp(v - max_v);
    sum += v;
  }

  const float inv_sum = 1.0f / sum;

  for (float &v : x) {
    v *= inv_sum;
  }
}

void add_inplace(std::span<float> x, std::span<const float> residual) {
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] += residual[i];
  }
}

} // namespace nakshatra

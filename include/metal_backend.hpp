#pragma once

#include <cstddef>
#include <memory>

namespace nakshatra {

enum class WeightFormat { F32, BF16 };

struct MetalJob {
  const float *w;
  const float *x;
  float *y;
  std::size_t out;
  std::size_t in;
};

class MetalMatvec {
public:
  MetalMatvec();
  ~MetalMatvec();

  MetalMatvec(const MetalMatvec &) = delete;
  MetalMatvec &operator=(const MetalMatvec &) = delete;

  bool ok() const;
  bool enabled() const;
  void set_enabled(bool on);

  void set_weight_format(WeightFormat fmt);
  WeightFormat weight_format() const;

  bool run(const float *w, const float *x, float *y, std::size_t out,
           std::size_t in);

  double bench(const float *w, const float *x, float *y, std::size_t out,
               std::size_t in, std::size_t warmup, std::size_t iters);

bool matvec(const float *w, const float *x, float *y, std::size_t out,
            std::size_t in);

bool matvec_batch(const MetalJob *jobs, std::size_t count);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  bool enabled_ = true;
};

MetalMatvec &global_metal();

} // namespace nakshatra

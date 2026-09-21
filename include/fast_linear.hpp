#pragma once

#include <cstddef>
#include <span>

#include "metal_backend.hpp"
#include "ops.hpp"

namespace nakshatra {

inline void fast_linear(const float *w, std::span<const float> x,
                        std::size_t out, std::size_t in, std::span<float> y) {
  MetalMatvec &metal = global_metal();

  if (metal.ok() && metal.enabled() && out * in >= 1000000 &&
      metal.matvec(w, x.data(), y.data(), out, in)) {
    return;
  }

  linear(x, std::span<const float>(w, out * in), out, in, y);
}

} // namespace nakshatra

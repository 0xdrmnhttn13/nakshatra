#include "rope.hpp"

#include <cmath>

namespace nakshatra {

std::vector<float> make_inv_freq(std::size_t head_dim, float theta) {
  const std::size_t half = head_dim / 2;
  std::vector<float> inv(half);

  for (std::size_t i = 0; i < half; ++i) {
    const float exponent =
        static_cast<float>(2 * i) / static_cast<float>(head_dim);
    inv[i] = 1.0f / std::pow(theta, exponent);
  }

  return inv;
}

void apply_rope(std::span<float> x, std::size_t position,
                std::span<const float> inv_freq) {
  const std::size_t dim = x.size();
  const std::size_t half = dim / 2;

  std::vector<float> old(x.begin(), x.end());

  for (std::size_t i = 0; i < half; ++i) {
    const float angle = static_cast<float>(position) * inv_freq[i];

    const float c = std::cos(angle);
    const float s = std::sin(angle);

    const float x1 = old[i];
    const float x2 = old[i + half];

    x[i] = x1 * c - x2 * s;
    x[i + half] = x2 * c + x1 * s;
  }
}

} // namespace nakshatra

#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace nakshatra {

std::vector<float> make_inv_freq(std::size_t head_dim, float theta);

void apply_rope(std::span<float> x, std::size_t position,
                std::span<const float> inv_freq);

} // namespace nakshatra

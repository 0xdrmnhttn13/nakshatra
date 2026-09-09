#pragma once

#include <cstddef>
#include <span>

namespace nakshatra {
float dot(const float *a, const float *b, std::size_t n);
void linear(std::span<const float> x, std::span<const float> weight,
            std::size_t out_feature, std::size_t in_features,
            std::span<float> y);

void gemma_rmsnorm(std::span<const float> x, std::span<const float> weight,
                   float eps, std::span<float> y);

float gelu_tanh(float x);

void gemma_mlp(std::span<const float> x, std::span<const float> gate_w,
               std::span<const float> up_w, std::span<const float> down_w,
               std::size_t hidden, std::size_t intermediate,
               std::span<float> y);

void softmax_inplace(std::span<float> x);

void add_inplace(std::span<float> x, std::span<const float> residual);

} // namespace nakshatra

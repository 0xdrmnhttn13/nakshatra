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

void normalize_heads(std::span<float> x, std::size_t num_heads,
                     std::size_t head_dim, std::span<const float> norm_weight,
                     float eps);

std::size_t kv_head_for_q_head(std::size_t q_head, std::size_t num_q_heads,
                               std::size_t num_kv_heads);

float attention_scale(float query_pre_attn_scalar);

void attend_one_head(std::span<const float> q, const float *key_base,
                     const float *value_base, std::size_t seq_begin,
                     std::size_t seq_end, std::size_t head_dim, float scale,
                     std::span<float> out);

} // namespace nakshatra

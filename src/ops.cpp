#include "ops.hpp"

#include "fast_linear.hpp"

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

  {
    nakshatra::MetalJob jobs[2]{
        {gate_w.data(), x.data(), gate.data(), intermediate, hidden},
        {up_w.data(), x.data(), up.data(), intermediate, hidden}};

    MetalMatvec &metal = global_metal();
    if (!(metal.ok() && metal.enabled() && metal.matvec_batch(jobs, 2))) {
      linear(x, gate_w, intermediate, hidden, gate);
      linear(x, up_w, intermediate, hidden, up);
    }
  }

  for (std::size_t i = 0; i < intermediate; ++i) {
    fused[i] = gelu_tanh(gate[i]) * up[i];
  }

  fast_linear(down_w.data(), fused, hidden, intermediate, y);
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

void normalize_heads(std::span<float> x, std::size_t num_heads,
                     std::size_t head_dim, std::span<const float> norm_weight,
                     float eps) {
  std::vector<float> temp(head_dim);
  for (std::size_t h = 0; h < num_heads; ++h) {
    auto head = x.subspan(h * head_dim, head_dim);
    gemma_rmsnorm(head, norm_weight, eps, temp);

    std::copy(temp.begin(), temp.end(), head.begin());
  }
}

std::size_t kv_head_for_q_head(std::size_t q_head, std::size_t num_q_heads,
                               std::size_t num_kv_heads) {
  const std::size_t group = num_q_heads / num_kv_heads;
  return q_head / group;
}

float attention_scale(float query_pre_attn_scalar) {
  return 1.0f / std::sqrt(query_pre_attn_scalar);
}

void attend_one_head(std::span<const float> q, const float *key_base,
                     const float *value_base, std::size_t seq_begin,
                     std::size_t seq_end, std::size_t head_dim, float scale,
                     std::span<float> out) {
  const std::size_t count = seq_end - seq_begin;
  std::vector<float> scores(count);

  for (std::size_t t = 0; t < count; ++t) {
    const float *k = key_base + (seq_begin + t) * head_dim;
    scores[t] = dot(q.data(), k, head_dim) * scale;
  }
  softmax_inplace(scores);
  std::fill(out.begin(), out.end(), 0.0f);
  for (std::size_t t = 0; t < count; ++t) {
    const float *v = value_base + (seq_begin + t) * head_dim;
    const float p = scores[t];
    for (std::size_t d = 0; d < head_dim; ++d) {
      out[d] += p * v[d];
    }
  }
}

} // namespace nakshatra

#pragma once
#include <cstddef>
#include <cstdint>

namespace nakshatra {
struct Gemma3Config {
  std::size_t vocab_size = 262144;
  std::size_t hidden_size = 1152;
  std::size_t intermediate_size = 6912;

  std::size_t num_layers = 26;
  std::size_t num_q_heads = 4;
  std::size_t num_kv_heads = 1;
  std::size_t head_dim = 256;

  std::size_t max_seq_len = 32768;
  std::size_t sliding_window = 512;
  std::size_t sliding_window_pattern = 6;

  float rms_eps = 1.0e-6f;
  float rope_theta_global = 1'000'000.0f;
  float rope_theta_local = 10'000.0f;

  float query_pre_attn_scalar = 256.0f;

  bool is_local_layer(std::size_t layer) const {
    return ((layer + 1) % sliding_window_pattern) != 0;
  }

  std::size_t q_per_kv() const { return num_q_heads / num_kv_heads; }
};
} // namespace nakshatra

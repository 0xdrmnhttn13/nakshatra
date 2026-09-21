#pragma once

#include "config.hpp"

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace nakshatra {
class KVCache {
public:
  KVCache(const Gemma3Config &cfg, std::size_t capacity)
      : cfg_(cfg), capacity_(capacity),
        k_(cfg.num_layers * cfg.num_kv_heads * capacity * cfg.head_dim),
        v_(k_.size()) {}

  float *key(std::size_t layer, std::size_t kv_head, std::size_t position) {
    return k_.data() + offset(layer, kv_head, position);
  }

  float *value(std::size_t layer, std::size_t kv_head, std::size_t position) {
    return v_.data() + offset(layer, kv_head, position);
  }

private:
  std::size_t offset(std::size_t layer, std::size_t kv_head,
                     std::size_t position) const {
    if (layer >= cfg_.num_layers || kv_head >= cfg_.num_kv_heads ||
        position >= capacity_) {
      throw std::out_of_range("KV cache index");
    }
    return ((layer * cfg_.num_kv_heads + kv_head) * capacity_ + position) *
           cfg_.head_dim;
  }
  Gemma3Config cfg_;
  std::size_t capacity_;
  std::vector<float> k_;
  std::vector<float> v_;
};

inline void write_kv(KVCache &cache, std::size_t layer, std::size_t kv_head,
                     std::size_t position, std::span<const float> k,
                     std::span<const float> v) {

  std::copy(k.begin(), k.end(), cache.key(layer, kv_head, position));

  std::copy(v.begin(), v.end(), cache.value(layer, kv_head, position));
}
} // namespace nakshatra

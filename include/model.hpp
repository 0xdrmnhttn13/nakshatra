#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "config.hpp"
#include "kv_cache.hpp"
#include "safetensors.hpp"
#include "weights.hpp"

namespace nakshatra {

void attention_decode_one_token(const Gemma3Config &cfg,
                                const AttentionWeights &w, KVCache &cache,
                                std::size_t layer, std::size_t position,
                                std::span<const float> x, std::span<float> y);

void decoder_layer_decode(const Gemma3Config &cfg, const LayerWeights &w,
                          KVCache &cache, std::size_t layer,
                          std::size_t position, std::span<float> x);

void embed_token(const Gemma3Config &cfg, const Tensor &embedding,
                 std::uint32_t token_id, std::span<float> x);

void compute_logits(const Gemma3Config &cfg, const Tensor &embedding,
                    std::span<const float> x, std::span<float> logits);

std::uint32_t argmax_token(std::span<const float> logits);

std::uint32_t decode_one(const Gemma3Config &cfg, const ModelWeights &model,
                         KVCache &cache, std::uint32_t token,
                         std::size_t position);

ModelWeights load_gemma_weights(const SafeTensorFile &file,
                                const Gemma3Config &cfg);

} // namespace nakshatra

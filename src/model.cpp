#include "model.hpp"

#include "config.hpp"
#include "fast_linear.hpp"
#include "kv_cache.hpp"
#include "ops.hpp"
#include "rope.hpp"

#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace nakshatra {

void attention_decode_one_token(const Gemma3Config &cfg,
                                const AttentionWeights &w,
                                KVCache &cache, std::size_t layer,
                                std::size_t position,
                                std::span<const float> x,
                                std::span<float> y) {
  const std::size_t q_dim = cfg.num_q_heads * cfg.head_dim;
  const std::size_t kv_dim = cfg.num_kv_heads * cfg.head_dim;

  std::vector<float> q(q_dim);
  std::vector<float> k(kv_dim);
  std::vector<float> v(kv_dim);

  {
    nakshatra::MetalJob jobs[3]{
        {w.q_proj.data(), x.data(), q.data(), q_dim, cfg.hidden_size},
        {w.k_proj.data(), x.data(), k.data(), kv_dim, cfg.hidden_size},
        {w.v_proj.data(), x.data(), v.data(), kv_dim, cfg.hidden_size}};

    nakshatra::MetalMatvec &metal = nakshatra::global_metal();
    if (!(metal.ok() && metal.enabled() &&
          metal.matvec_batch(jobs, 3))) {
      nakshatra::linear(x, w.q_proj.span(), q_dim, cfg.hidden_size, q);
      nakshatra::linear(x, w.k_proj.span(), kv_dim, cfg.hidden_size, k);
      nakshatra::linear(x, w.v_proj.span(), kv_dim, cfg.hidden_size, v);
    }
  }  normalize_heads(q, cfg.num_q_heads, cfg.head_dim, w.q_norm.span(),
                  cfg.rms_eps);
  normalize_heads(k, cfg.num_kv_heads, cfg.head_dim, w.k_norm.span(),
                  cfg.rms_eps);

  const float theta =
      cfg.is_local_layer(layer) ? cfg.rope_theta_local : cfg.rope_theta_global;

  const auto inv_freq = make_inv_freq(cfg.head_dim, theta);

  for (std::size_t h = 0; h < cfg.num_q_heads; ++h) {
    apply_rope(std::span<float>(q.data() + h * cfg.head_dim, cfg.head_dim),
               position, inv_freq);
  }

  for (std::size_t h = 0; h < cfg.num_kv_heads; ++h) {
    apply_rope(std::span<float>(k.data() + h * cfg.head_dim, cfg.head_dim),
               position, inv_freq);

    write_kv(cache, layer, h, position,
             std::span<const float>(k.data() + h * cfg.head_dim, cfg.head_dim),
             std::span<const float>(v.data() + h * cfg.head_dim, cfg.head_dim));
  }

  std::vector<float> concat(q_dim);

  const std::size_t begin = cfg.is_local_layer(layer)
                                ? (position + 1 > cfg.sliding_window
                                       ? position + 1 - cfg.sliding_window
                                       : 0)
                                : 0;

  const std::size_t end = position + 1;

  const float scale = attention_scale(cfg.query_pre_attn_scalar);

  for (std::size_t qh = 0; qh < cfg.num_q_heads; ++qh) {

    const std::size_t kvh =
        kv_head_for_q_head(qh, cfg.num_q_heads, cfg.num_kv_heads);

    std::span<const float> q_head(q.data() + qh * cfg.head_dim, cfg.head_dim);

    std::span<float> out_head(concat.data() + qh * cfg.head_dim, cfg.head_dim);

    const float *key_base = cache.key(layer, kvh, 0);

    const float *value_base = cache.value(layer, kvh, 0);

    attend_one_head(q_head, key_base, value_base, begin, end, cfg.head_dim,
                    scale, out_head);
  }

  fast_linear(w.o_proj.data(), concat, cfg.hidden_size, q_dim, y);
}

void decoder_layer_decode(const Gemma3Config &cfg, const LayerWeights &w,
                          KVCache &cache, std::size_t layer,
                          std::size_t position, std::span<float> x) {
  std::vector<float> residual(x.begin(), x.end());

  std::vector<float> normed(cfg.hidden_size);
  std::vector<float> attn(cfg.hidden_size);
  std::vector<float> post(cfg.hidden_size);

  gemma_rmsnorm(x, w.input_layernorm.span(), cfg.rms_eps, normed);

  attention_decode_one_token(cfg, w.attn, cache, layer, position, normed, attn);

  gemma_rmsnorm(attn, w.post_attention_layernorm.span(), cfg.rms_eps, post);

  for (std::size_t i = 0; i < cfg.hidden_size; ++i) {
    x[i] = residual[i] + post[i];
  }

  residual.assign(x.begin(), x.end());

  gemma_rmsnorm(x, w.pre_feedforward_layernorm.span(), cfg.rms_eps, normed);

  std::vector<float> mlp_out(cfg.hidden_size);

  gemma_mlp(normed, w.mlp.gate_proj.span(), w.mlp.up_proj.span(),
            w.mlp.down_proj.span(), cfg.hidden_size, cfg.intermediate_size,
            mlp_out);

  gemma_rmsnorm(mlp_out, w.post_feedforward_layernorm.span(), cfg.rms_eps,
                post);

  for (std::size_t i = 0; i < cfg.hidden_size; ++i) {
    x[i] = residual[i] + post[i];
  }
}

void embed_token(const Gemma3Config &cfg, const Tensor &embedding,
                 std::uint32_t token_id, std::span<float> x) {
  if (token_id >= cfg.vocab_size) {
    throw std::out_of_range("token id");
  }

  const float *row =
      embedding.data() +
      static_cast<std::size_t>(token_id) * cfg.hidden_size;

  const float scale =
      std::sqrt(static_cast<float>(cfg.hidden_size));

  for (std::size_t i = 0; i < cfg.hidden_size; ++i) {
    x[i] = row[i] * scale;
  }
}

void compute_logits(const Gemma3Config &cfg, const Tensor &embedding,
                    std::span<const float> x, std::span<float> logits) {
  fast_linear(embedding.data(), x, cfg.vocab_size, cfg.hidden_size, logits);
}

std::uint32_t argmax_token(std::span<const float> logits) {
  std::size_t best = 0;

  for (std::size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > logits[best]) {
      best = i;
    }
  }

  return static_cast<std::uint32_t>(best);
}

std::uint32_t decode_one(const Gemma3Config &cfg, const ModelWeights &model,
                         KVCache &cache, std::uint32_t token,
                         std::size_t position) {
  std::vector<float> x(cfg.hidden_size);

  embed_token(cfg, model.embedding, token, x);

  for (std::size_t layer = 0; layer < cfg.num_layers; ++layer) {
    decoder_layer_decode(cfg, model.layers[layer], cache, layer, position, x);
  }

  std::vector<float> normed(cfg.hidden_size);

  gemma_rmsnorm(x, model.final_norm.span(), cfg.rms_eps, normed);

  std::vector<float> logits(cfg.vocab_size);

  compute_logits(cfg, model.embedding, normed, logits);

  return argmax_token(logits);
}

ModelWeights load_gemma_weights(const SafeTensorFile &file,
                                const Gemma3Config &cfg) {
  ModelWeights m;

  m.embedding =
      materialize_bf16(file, "model.embed_tokens.weight");
  m.final_norm = materialize_bf16(file, "model.norm.weight");

  m.layers.resize(cfg.num_layers);

  for (std::size_t i = 0; i < cfg.num_layers; ++i) {
    LayerWeights &w = m.layers[i];
    const std::string p = "model.layers." + std::to_string(i) + ".";

    w.input_layernorm =
        materialize_bf16(file, p + "input_layernorm.weight");
    w.post_attention_layernorm =
        materialize_bf16(file, p + "post_attention_layernorm.weight");
    w.pre_feedforward_layernorm =
        materialize_bf16(file, p + "pre_feedforward_layernorm.weight");
    w.post_feedforward_layernorm =
        materialize_bf16(file, p + "post_feedforward_layernorm.weight");

    w.attn.q_proj =
        materialize_bf16(file, p + "self_attn.q_proj.weight");
    w.attn.k_proj =
        materialize_bf16(file, p + "self_attn.k_proj.weight");
    w.attn.v_proj =
        materialize_bf16(file, p + "self_attn.v_proj.weight");
    w.attn.o_proj =
        materialize_bf16(file, p + "self_attn.o_proj.weight");
    w.attn.q_norm =
        materialize_bf16(file, p + "self_attn.q_norm.weight");
    w.attn.k_norm =
        materialize_bf16(file, p + "self_attn.k_norm.weight");

    w.mlp.gate_proj = materialize_bf16(file, p + "mlp.gate_proj.weight");
    w.mlp.up_proj = materialize_bf16(file, p + "mlp.up_proj.weight");
    w.mlp.down_proj = materialize_bf16(file, p + "mlp.down_proj.weight");
  }

  return m;
}

} // namespace nakshatra

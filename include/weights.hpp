#pragma once

#include <vector>

#include "tensor.hpp"

namespace nakshatra {

struct AttentionWeights {
  Tensor q_proj;
  Tensor k_proj;
  Tensor v_proj;
  Tensor o_proj;

  Tensor q_norm;
  Tensor k_norm;
};

struct MLPWeights {
  Tensor gate_proj;
  Tensor up_proj;
  Tensor down_proj;
};

struct LayerWeights {
  Tensor input_layernorm;
  Tensor post_attention_layernorm;
  Tensor pre_feedforward_layernorm;
  Tensor post_feedforward_layernorm;

  AttentionWeights attn;
  MLPWeights mlp;
};

struct ModelWeights {
  Tensor embedding;
  std::vector<LayerWeights> layers;
  Tensor final_norm;
};

} // namespace nakshatra

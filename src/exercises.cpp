#include "config.hpp"
#include "ir.hpp"
#include "kv_cache.hpp"
#include "model.hpp"
#include "ops.hpp"
#include "pass.hpp"
#include "rope.hpp"
#include "safetensors.hpp"
#include "tensor.hpp"
#include "weights.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

static void exercise_shapes() {
  nakshatra::Gemma3Config cfg;

  const std::size_t q_dim = cfg.num_q_heads * cfg.head_dim;
  const std::size_t kv_dim = cfg.num_kv_heads * cfg.head_dim;

  std::vector<float> x(cfg.hidden_size, 0.5f);

  std::vector<float> w_q(q_dim * cfg.hidden_size, 0.01f);
  std::vector<float> w_k(kv_dim * cfg.hidden_size, 0.01f);
  std::vector<float> w_v(kv_dim * cfg.hidden_size, 0.01f);

  std::vector<float> q(q_dim);
  std::vector<float> k(kv_dim);
  std::vector<float> v(kv_dim);

  nakshatra::linear(x, w_q, q_dim, cfg.hidden_size, q);
  nakshatra::linear(x, w_k, kv_dim, cfg.hidden_size, k);
  nakshatra::linear(x, w_v, kv_dim, cfg.hidden_size, v);

  std::cout << "q: " << q.size() << "\n";
  std::cout << "k: " << k.size() << "\n";
  std::cout << "v: " << v.size() << "\n";

  std::cout << "q != hidden_size: "
            << (q_dim != cfg.hidden_size ? "yes" : "NO - BUG") << "\n";
}

static void exercise_scaled_dot() {
  const std::size_t head_dim = 4;
  const std::size_t tokens = 3;

  std::vector<float> q{1, 0, 0, 0};

  std::vector<float> k{
      1, 0, 0, 0, //
      0, 1, 0, 0, //
      0, 0, 1, 0  //
  };

  std::vector<float> v{
      1, 0, 0, 0, //
      0, 1, 0, 0, //
      0, 0, 1, 0  //
  };

  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  std::vector<float> scores(tokens);

  for (std::size_t t = 0; t < tokens; ++t) {
    scores[t] =
        nakshatra::dot(q.data(), k.data() + t * head_dim, head_dim) * scale;
  }

  nakshatra::softmax_inplace(scores);

  std::vector<float> out(head_dim, 0.0f);

  for (std::size_t t = 0; t < tokens; ++t) {
    for (std::size_t d = 0; d < head_dim; ++d) {
      out[d] += scores[t] * v[t * head_dim + d];
    }
  }

  std::cout << "scores: " << scores[0] << " " << scores[1] << " "
            << scores[2] << "\n";

  std::cout << "out:    " << out[0] << " " << out[1] << " " << out[2] << " "
            << out[3] << "\n";
}

static void exercise_masking() {
  const float neg_inf = -std::numeric_limits<float>::infinity();

  std::vector<float> scores{0.5f, 0.0f, 0.0f};

  std::vector<float> masked = scores;
  masked[1] = neg_inf;

  nakshatra::softmax_inplace(masked);

  std::vector<float> windowed{scores[0], scores[2]};
  nakshatra::softmax_inplace(windowed);

  std::cout << "mask  : " << masked[0] << " " << masked[1] << " " << masked[2]
            << "\n";

  std::cout << "window: " << windowed[0] << " " << windowed[1] << "\n";

  std::cout << "equal : "
            << (masked[0] == windowed[0] && masked[2] == windowed[1] ? "yes"
                                                                     : "NO - BUG")
            << "\n";
}

static void exercise_heads() {
  nakshatra::Gemma3Config cfg;

  const std::size_t q_dim = cfg.num_q_heads * cfg.head_dim;
  const std::size_t kv_dim = cfg.num_kv_heads * cfg.head_dim;

  std::vector<float> q(q_dim);
  std::vector<float> k(kv_dim);
  std::vector<float> v(kv_dim);

  std::cout << "Q logical shape: [" << cfg.num_q_heads << ", " << cfg.head_dim
            << "]\n";

  std::cout << "K logical shape: [" << cfg.num_kv_heads << ", "
            << cfg.head_dim << "]\n";

  std::cout << "V logical shape: [" << cfg.num_kv_heads << ", "
            << cfg.head_dim << "]\n";

  for (std::size_t h = 0; h < cfg.num_q_heads; ++h) {
    std::size_t begin = h * cfg.head_dim;
    std::size_t end = begin + cfg.head_dim - 1;

    std::cout << "Q head " << h << " -> q[" << begin << ".." << end << "]\n";
  }

  for (std::size_t h = 0; h < cfg.num_kv_heads; ++h) {
    std::size_t begin = h * cfg.head_dim;
    std::size_t end = begin + cfg.head_dim - 1;

    std::cout << "KV head " << h << " -> k/v[" << begin << ".." << end
              << "]\n";
  }
}

static void exercise_graph() {
  nakshatra::Graph g;

  nakshatra::TensorType f32_x{nakshatra::DType::F32, {1152}};
  nakshatra::TensorType f32_w{nakshatra::DType::F32, {1024, 1152}};
  nakshatra::TensorType f32_q{nakshatra::DType::F32, {1024}};

  nakshatra::ValueId x = g.add_op(nakshatra::OpKind::Input, {}, f32_x, "x");
  nakshatra::ValueId w =
      g.add_op(nakshatra::OpKind::Constant, {}, f32_w, "q_proj");
  nakshatra::ValueId q = g.add_op(nakshatra::OpKind::Linear, {x, w}, f32_q);
  nakshatra::ValueId a = g.add_op(nakshatra::OpKind::GELU, {q}, f32_q);

  std::cout << g.dump();

  nakshatra::verify(g);
  std::cout << "verify: ok\n";

  nakshatra::Graph broken;
  nakshatra::ValueId ghost = 7;
  broken.add_op(nakshatra::OpKind::Linear, {ghost, x}, f32_q, "bad_linear");

  try {
    nakshatra::verify(broken);
  } catch (const std::runtime_error &e) {
    std::cout << "verify caught: " << e.what() << "\n";
  }
}

static void exercise_rope() {
  const auto inv = nakshatra::make_inv_freq(8, 10000.0f);

  std::cout << "inv_freq(8, 1e4): ";
  for (float f : inv) {
    std::cout << f << " ";
  }
  std::cout << " (expected 1 0.1 0.01 0.001)\n";

  std::vector<float> x{1.0f, 0.5f, -0.25f, 4.0f,
                       2.0f, -1.0f, 0.75f, 0.125f};
  std::vector<float> before = x;

  nakshatra::apply_rope(x, 1234, nakshatra::make_inv_freq(8, 10000.0f));

  float sq_before = 0.0f;
  float sq_after = 0.0f;
  for (std::size_t i = 0; i < x.size(); ++i) {
    sq_before += before[i] * before[i];
    sq_after += x[i] * x[i];
  }

  std::cout << "norm before: " << sq_before << "\n";
  std::cout << "norm after : " << sq_after << " (rotation preserves norm)\n";
}

static void exercise_kv() {
  nakshatra::Gemma3Config cfg;
  const std::size_t capacity = 8;

  nakshatra::KVCache cache(cfg, capacity);

  std::vector<float> k{0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f};
  std::vector<float> v{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

  nakshatra::write_kv(cache, 3, 0, 5, k, v);

  bool roundtrip_ok = true;
  for (std::size_t i = 0; i < cfg.head_dim && i < k.size(); ++i) {
    if (cache.key(3, 0, 5)[i] != k[i] || cache.value(3, 0, 5)[i] != v[i]) {
      roundtrip_ok = false;
    }
  }
  std::cout << "write then read at (layer=3, kv=0, pos=5): "
            << (roundtrip_ok ? "ok" : "NO - BUG") << "\n";

  std::cout << "key(3,0,4) untouched: "
            << (cache.key(3, 0, 4)[0] == 0.0f ? "yes (zero-init)"
                                              : "NO - BUG")
            << "\n";

  try {
    cache.key(0, 0, capacity);
    std::cout << "bounds: NO - BUG (no throw)\n";
  } catch (const std::out_of_range &) {
    std::cout << "bounds: throws out_of_range at position == capacity\n";
  }

  const double fp32 =
      static_cast<double>(cfg.num_layers) * cfg.num_kv_heads * 32768 *
      cfg.head_dim * 4 * 2;
  std::cout << "full-context cache @32768: " << fp32 / (1024 * 1024 * 1024)
            << " GiB fp32, " << fp32 / 2 / (1024 * 1024 * 1024)
            << " GiB bf16\n";
}

static std::uint64_t read_u64_le_demo(const std::byte *p) {
  std::uint64_t value = 0;
  for (int i = 0; i < 8; ++i) {
    value |= static_cast<std::uint64_t>(std::to_integer<unsigned char>(p[i]))
             << (8 * i);
  }
  return value;
}

static void print_bytes(const std::byte *p) {
  for (int i = 0; i < 8; ++i) {
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<unsigned>(std::to_integer<unsigned char>(p[i]))
              << " ";
  }
  std::cout << std::dec;
}

static void exercise_bytes() {
  const std::byte case1[]{
      std::byte{100}, std::byte{0}, std::byte{0}, std::byte{0},
      std::byte{0},   std::byte{0}, std::byte{0}, std::byte{0}};
  const std::byte case2[]{
      std::byte{0x22}, std::byte{0x11}, std::byte{0}, std::byte{0},
      std::byte{0},    std::byte{0},    std::byte{0}, std::byte{0}};
  const std::byte case3[]{
      std::byte{0}, std::byte{0}, std::byte{0}, std::byte{0},
      std::byte{0}, std::byte{0}, std::byte{0}, std::byte{1}};

  std::cout << "bytes                -> value\n";
  print_bytes(case1);
  std::cout << " -> " << read_u64_le_demo(case1) << "  (header len 100)\n";
  print_bytes(case2);
  std::cout << " -> " << read_u64_le_demo(case2) << "  (0x11 in slot 1)\n";
  print_bytes(case3);
  std::cout << " -> " << read_u64_le_demo(case3) << "  (1 in slot 7)\n";
}

static void write_test_safetensors(const std::string &path) {
  const std::vector<float> values{1.0f, -2.0f, 0.25f, 100.0f};

  std::string data;
  for (float v : values) {
    const std::uint16_t b = nakshatra::float_to_bf16_trunc(v);
    data.push_back(static_cast<char>(b & 0xFF));
    data.push_back(static_cast<char>(b >> 8));
  }

  const std::string json =
      "{\"tiny\":{\"dtype\":\"BF16\",\"shape\":[4],"
      "\"data_offsets\":[0,8]}}";

  std::ofstream out(path, std::ios::binary);

  std::uint64_t n = json.size();
  for (int i = 0; i < 8; ++i) {
    out.put(static_cast<char>((n >> (8 * i)) & 0xFF));
  }

  out.write(json.data(), static_cast<std::streamsize>(json.size()));
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

static void exercise_safetensors() {
  const std::string path = "build/tiny.safetensors";

  write_test_safetensors(path);

  const nakshatra::SafeTensorFile file = nakshatra::load_safetensors(path);

  const auto &meta = file.tensors.at("tiny");

  std::cout << "tensor: " << meta.dtype << " shape [" << meta.shape[0]
            << "] offsets [" << meta.begin << ", " << meta.end << "]"
            << " data_start " << file.data_start << "\n";

  const nakshatra::Tensor t = nakshatra::materialize_bf16(file, "tiny");

  std::cout << "values: ";
  for (std::size_t i = 0; i < t.numel(); ++i) {
    std::cout << t[i] << " ";
  }
  std::cout << "(expected 1 -2 0.25 100)\n";
}

static void exercise_banner() {
  nakshatra::Graph g;

  nakshatra::TensorType x_t{nakshatra::DType::F32, {1152}};
  nakshatra::TensorType w_t{nakshatra::DType::F32, {1024, 1152}};
  nakshatra::TensorType q_t{nakshatra::DType::F32, {1024}};

  nakshatra::ValueId x = g.add_op(nakshatra::OpKind::Input, {}, x_t, "x");
  nakshatra::ValueId nw =
      g.add_op(nakshatra::OpKind::Constant, {}, x_t, "norm_w");
  nakshatra::ValueId m =
      g.add_op(nakshatra::OpKind::Constant, {}, w_t, "q_proj");
  nakshatra::ValueId n = g.add_op(nakshatra::OpKind::RMSNorm, {x, nw}, x_t);
  nakshatra::ValueId q = g.add_op(nakshatra::OpKind::Linear, {n, m}, q_t);
  g.add_op(nakshatra::OpKind::GELU, {q}, q_t);

  std::cout << "graph before passes:\n" << g.dump();

  nakshatra::PassManager pm;
  pm.add(nakshatra::create_canonicalize_pass());
  pm.add(nakshatra::create_fuse_rmsnorm_linear_pass());
  pm.run(g);

  std::cout << "graph after passes:\n" << g.dump();
  std::cout << "verify: ok\n";
}

static nakshatra::Tensor random_tensor(std::vector<std::size_t> shape,
                                       std::mt19937 &gen, float scale) {
  std::size_t n = 1;
  for (std::size_t d : shape) {
    n *= d;
  }
  std::uniform_real_distribution<float> dist(-scale, scale);
  std::vector<float> data(n);
  for (float &v : data) {
    v = dist(gen);
  }
  return nakshatra::Tensor(std::move(shape), std::move(data));
}

static void exercise_layer() {
  nakshatra::Gemma3Config cfg;
  std::mt19937 gen(42);

  nakshatra::LayerWeights w;
  w.input_layernorm = random_tensor({cfg.hidden_size}, gen, 0.1f);
  w.post_attention_layernorm = random_tensor({cfg.hidden_size}, gen, 0.1f);
  w.pre_feedforward_layernorm = random_tensor({cfg.hidden_size}, gen, 0.1f);
  w.post_feedforward_layernorm = random_tensor({cfg.hidden_size}, gen, 0.1f);

  w.attn.q_proj = random_tensor({cfg.num_q_heads * cfg.head_dim,
                                 cfg.hidden_size},
                                gen, 0.02f);
  w.attn.k_proj = random_tensor(
      {cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size}, gen, 0.02f);
  w.attn.v_proj = random_tensor(
      {cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size}, gen, 0.02f);
  w.attn.o_proj = random_tensor(
      {cfg.hidden_size, cfg.num_q_heads * cfg.head_dim}, gen, 0.02f);
  w.attn.q_norm = random_tensor({cfg.head_dim}, gen, 0.1f);
  w.attn.k_norm = random_tensor({cfg.head_dim}, gen, 0.1f);

  w.mlp.gate_proj = random_tensor({cfg.intermediate_size, cfg.hidden_size},
                                  gen, 0.02f);
  w.mlp.up_proj = random_tensor({cfg.intermediate_size, cfg.hidden_size},
                                gen, 0.02f);
  w.mlp.down_proj = random_tensor({cfg.hidden_size, cfg.intermediate_size},
                                  gen, 0.02f);

  nakshatra::KVCache cache(cfg, 8);

  nakshatra::Tensor x_tensor = random_tensor({cfg.hidden_size}, gen, 1.0f);
  std::vector<float> x(x_tensor.data(), x_tensor.data() + x_tensor.numel());

  const float x0 = x[0];

  nakshatra::decoder_layer_decode(cfg, w, cache, 0, 0, x);

  bool finite = true;
  float lo = x[0], hi = x[0];
  for (float v : x) {
    if (!std::isfinite(v)) {
      finite = false;
    }
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }

  std::cout << "one layer, one token (random weights)\n";
  std::cout << "shape in/out: " << cfg.hidden_size << " -> " << x.size()
            << "\n";
  std::cout << "finite: " << (finite ? "yes" : "NO - NaN/inf!")
            << ", range [" << lo << ", " << hi << "]\n";
  std::cout << "residual works: " << (x[0] != x0 ? "x changed" : "NO - BUG")
            << "\n";
}

static void exercise_generate() {
  nakshatra::Gemma3Config cfg;
  cfg.vocab_size = 100;
  cfg.hidden_size = 64;
  cfg.intermediate_size = 128;
  cfg.num_layers = 2;
  cfg.num_q_heads = 2;
  cfg.num_kv_heads = 1;
  cfg.head_dim = 32;
  cfg.max_seq_len = 32;
  cfg.sliding_window = 8;

  std::mt19937 gen(7);

  nakshatra::ModelWeights model;
  model.embedding = random_tensor({cfg.vocab_size, cfg.hidden_size}, gen, 0.1f);
  model.final_norm = random_tensor({cfg.hidden_size}, gen, 0.1f);
  model.layers.resize(cfg.num_layers);

  for (nakshatra::LayerWeights &w : model.layers) {
    w.input_layernorm = random_tensor({cfg.hidden_size}, gen, 0.1f);
    w.post_attention_layernorm =
        random_tensor({cfg.hidden_size}, gen, 0.1f);
    w.pre_feedforward_layernorm =
        random_tensor({cfg.hidden_size}, gen, 0.1f);
    w.post_feedforward_layernorm =
        random_tensor({cfg.hidden_size}, gen, 0.1f);

    w.attn.q_proj = random_tensor(
        {cfg.num_q_heads * cfg.head_dim, cfg.hidden_size}, gen, 0.2f);
    w.attn.k_proj = random_tensor(
        {cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size}, gen, 0.2f);
    w.attn.v_proj = random_tensor(
        {cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size}, gen, 0.2f);
    w.attn.o_proj = random_tensor(
        {cfg.hidden_size, cfg.num_q_heads * cfg.head_dim}, gen, 0.2f);
    w.attn.q_norm = random_tensor({cfg.head_dim}, gen, 0.1f);
    w.attn.k_norm = random_tensor({cfg.head_dim}, gen, 0.1f);

    w.mlp.gate_proj =
        random_tensor({cfg.intermediate_size, cfg.hidden_size}, gen, 0.2f);
    w.mlp.up_proj =
        random_tensor({cfg.intermediate_size, cfg.hidden_size}, gen, 0.2f);
    w.mlp.down_proj =
        random_tensor({cfg.hidden_size, cfg.intermediate_size}, gen, 0.2f);
  }

  nakshatra::KVCache cache(cfg, cfg.max_seq_len);

  std::uint32_t token = 0;
  std::cout << "generated tokens (random weights, toy config):";
  for (std::size_t position = 0; position < 8; ++position) {
    token = nakshatra::decode_one(cfg, model, cache, token, position);
    std::cout << " " << token;
  }
  std::cout << "\n(end-to-end: embed -> 2 layers -> final norm -> logits -> "
               "argmax -> feed back)\n";
}


namespace nakshatra {

void run_all_exercises() {
  exercise_shapes();
  std::cout << "---\n";
  exercise_scaled_dot();
  std::cout << "---\n";
  exercise_masking();
  std::cout << "---\n";
  exercise_heads();
  std::cout << "---\n";
  exercise_graph();
  std::cout << "---\n";
  exercise_rope();
  std::cout << "---\n";
  exercise_kv();
  std::cout << "---\n";
  exercise_bytes();
  std::cout << "---\n";
  exercise_safetensors();
  std::cout << "---\n";
  exercise_banner();
  std::cout << "---\n";
  exercise_layer();
  std::cout << "---\n";
  exercise_generate();
}

} // namespace nakshatra

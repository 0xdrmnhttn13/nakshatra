#include "config.hpp"
#include "ir.hpp"
#include "ops.hpp"

#include <cmath>
#include <iostream>
#include <limits>
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

int main() {
  exercise_shapes();
  std::cout << "---\n";
  exercise_scaled_dot();
  std::cout << "---\n";
  exercise_masking();
  std::cout << "---\n";
  exercise_heads();
  std::cout << "---\n";
  exercise_graph();
  return 0;
}

#include "ir.hpp"
#include "ops.hpp"
#include "pass.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const char *what) {
  std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) {
    ++g_failures;
  }
}

std::size_t count_kind(const nakshatra::Graph &g, nakshatra::OpKind kind) {
  std::size_t n = 0;
  for (const nakshatra::Op &op : g.ops()) {
    if (op.kind == kind) {
      ++n;
    }
  }
  return n;
}

const nakshatra::Op *find_kind(const nakshatra::Graph &g,
                               nakshatra::OpKind kind) {
  for (const nakshatra::Op &op : g.ops()) {
    if (op.kind == kind) {
      return &op;
    }
  }
  return nullptr;
}

void set_int_attr(nakshatra::Graph &g, const char *key, std::int64_t value) {
  g.ops().back().int_attrs[key] = value;
}

void run_pipeline(nakshatra::Graph &g, bool with_fusion, const char *label) {
  std::printf("\n--- %s ---\n[sebelum]\n%s", label, g.dump().c_str());
  try {
    nakshatra::PassManager pm;
    pm.add(nakshatra::create_canonicalize_pass());
    if (with_fusion) {
      pm.add(nakshatra::create_fuse_rmsnorm_linear_pass());
    }
    pm.run(g);
    std::printf("[sesudah]\n%s", g.dump().c_str());
    check(true, label);
  } catch (const std::exception &error) {
    std::printf("    (%s)\n", error.what());
    check(false, label);
  }
}

void test_add_zero_redirects_every_use() {
  nakshatra::Graph g;
  const nakshatra::TensorType v{nakshatra::DType::F32, {8}};

  const nakshatra::ValueId x =
      g.add_op(nakshatra::OpKind::Input, {}, v, "x");
  const nakshatra::ValueId zero =
      g.add_op(nakshatra::OpKind::Constant, {}, v, "zero");
  set_int_attr(g, "fill", 0);
  const nakshatra::ValueId a =
      g.add_op(nakshatra::OpKind::Add, {x, zero}, v);
  g.add_op(nakshatra::OpKind::GELU, {a}, v);
  g.add_op(nakshatra::OpKind::Multiply, {a, x}, v);

  run_pipeline(g, false, "add-zero leaves valid IR");

  check(count_kind(g, nakshatra::OpKind::Add) == 0,
        "add-zero removes the add");
  check(count_kind(g, nakshatra::OpKind::Constant) == 0,
        "add-zero drops the dead zero constant");

  const nakshatra::Op *gelu = find_kind(g, nakshatra::OpKind::GELU);
  const nakshatra::Op *mul = find_kind(g, nakshatra::OpKind::Multiply);
  check(gelu != nullptr && gelu->inputs.size() == 1 && gelu->inputs[0] == x,
        "add-zero redirects the gelu use");
  check(mul != nullptr && mul->inputs.size() == 2 && mul->inputs[0] == x &&
            mul->inputs[1] == x,
        "add-zero redirects the multiply use");
}

void test_multiply_one_redirects_every_use() {
  nakshatra::Graph g;
  const nakshatra::TensorType v{nakshatra::DType::F32, {8}};

  const nakshatra::ValueId x =
      g.add_op(nakshatra::OpKind::Input, {}, v, "x");
  const nakshatra::ValueId one =
      g.add_op(nakshatra::OpKind::Constant, {}, v, "one");
  set_int_attr(g, "fill", 1);
  const nakshatra::ValueId r =
      g.add_op(nakshatra::OpKind::Multiply, {x, one}, v);
  g.add_op(nakshatra::OpKind::Add, {r, x}, v);

  run_pipeline(g, false, "multiply-one leaves valid IR");

  check(count_kind(g, nakshatra::OpKind::Multiply) == 0,
        "multiply-one removes the multiply");
  check(count_kind(g, nakshatra::OpKind::Constant) == 0,
        "multiply-one drops the dead one constant");

  const nakshatra::Op *add = find_kind(g, nakshatra::OpKind::Add);
  check(add != nullptr && add->inputs.size() == 2 && add->inputs[0] == x &&
            add->inputs[1] == x,
        "multiply-one redirects the add use");
}

void test_rope_zero_folded_nonzero_preserved() {
  nakshatra::Graph g;
  const nakshatra::TensorType v{nakshatra::DType::F32, {8}};

  const nakshatra::ValueId x =
      g.add_op(nakshatra::OpKind::Input, {}, v, "x");

  const nakshatra::ValueId r0 = g.add_op(nakshatra::OpKind::RoPE, {x}, v);
  set_int_attr(g, "position", 0);
  g.add_op(nakshatra::OpKind::GELU, {r0}, v);

  const nakshatra::ValueId r5 = g.add_op(nakshatra::OpKind::RoPE, {x}, v);
  set_int_attr(g, "position", 5);
  g.add_op(nakshatra::OpKind::GELU, {r5}, v);

  run_pipeline(g, false, "rope fold leaves valid IR");

  check(count_kind(g, nakshatra::OpKind::RoPE) == 1,
        "rope at position 0 folds, exactly one rope remains");

  const nakshatra::Op *rope = find_kind(g, nakshatra::OpKind::RoPE);
  check(rope != nullptr && rope->inputs.size() == 1 && rope->inputs[0] == x &&
            rope->int_attrs.at("position") == 5,
        "rope at nonzero position is preserved");

  bool folded_use = false;
  for (const nakshatra::Op &op : g.ops()) {
    if (op.kind == nakshatra::OpKind::GELU && op.inputs.size() == 1 &&
        op.inputs[0] == x) {
      folded_use = true;
    }
  }
  check(folded_use, "rope at position 0 redirects its use to x");
}

void test_fusion_rejects_multi_use_norm() {
  nakshatra::Graph g;
  const nakshatra::TensorType v8{nakshatra::DType::F32, {8}};
  const nakshatra::TensorType v16{nakshatra::DType::F32, {16, 8}};

  const nakshatra::ValueId x =
      g.add_op(nakshatra::OpKind::Input, {}, v8, "x");
  const nakshatra::ValueId w =
      g.add_op(nakshatra::OpKind::Constant, {}, v8, "norm_weight");
  const nakshatra::ValueId m =
      g.add_op(nakshatra::OpKind::Constant, {}, v16, "matrix");

  const nakshatra::ValueId n =
      g.add_op(nakshatra::OpKind::RMSNorm, {x, w}, v8);
  g.add_op(nakshatra::OpKind::Linear, {n, m}, v16);
  g.add_op(nakshatra::OpKind::GELU, {n}, v8);

  run_pipeline(g, true, "multi-use fusion attempt leaves valid IR");

  check(find_kind(g, nakshatra::OpKind::FusedRMSNormLinear) == nullptr,
        "fusion rejects a multi-use norm result");

  const nakshatra::Op *linear = find_kind(g, nakshatra::OpKind::Linear);
  check(linear != nullptr && linear->inputs.size() == 2 &&
            linear->inputs[0] == n && linear->inputs[1] == m,
        "the unfused linear is preserved");
}

void test_fusion_happy_path() {
  nakshatra::Graph g;
  const nakshatra::TensorType v8{nakshatra::DType::F32, {8}};
  const nakshatra::TensorType v16{nakshatra::DType::F32, {16, 8}};

  const nakshatra::ValueId x =
      g.add_op(nakshatra::OpKind::Input, {}, v8, "x");
  const nakshatra::ValueId w =
      g.add_op(nakshatra::OpKind::Constant, {}, v8, "norm_weight");
  const nakshatra::ValueId m =
      g.add_op(nakshatra::OpKind::Constant, {}, v16, "matrix");

  const nakshatra::ValueId n =
      g.add_op(nakshatra::OpKind::RMSNorm, {x, w}, v8);
  const nakshatra::ValueId y =
      g.add_op(nakshatra::OpKind::Linear, {n, m}, v16);
  g.add_op(nakshatra::OpKind::GELU, {y}, v16);

  run_pipeline(g, true, "fusion leaves valid IR");

  const nakshatra::Op *fused =
      find_kind(g, nakshatra::OpKind::FusedRMSNormLinear);
  check(fused != nullptr && fused->result == y,
        "fusion produces FusedRMSNormLinear keeping the linear result");
  check(fused != nullptr && fused->inputs.size() == 3 &&
            fused->inputs[0] == x && fused->inputs[1] == w &&
            fused->inputs[2] == m,
        "fused op takes (x, norm_weight, matrix)");

  check(count_kind(g, nakshatra::OpKind::RMSNorm) == 0 &&
            count_kind(g, nakshatra::OpKind::Linear) == 0,
        "fusion removes the original rmsnorm and linear");

  const nakshatra::Op *gelu = find_kind(g, nakshatra::OpKind::GELU);
  check(gelu != nullptr && gelu->inputs.size() == 1 && gelu->inputs[0] == y,
        "downstream use still points at the fused result");
}

void test_canonicalize_idempotent() {
  nakshatra::Graph g;
  const nakshatra::TensorType v{nakshatra::DType::F32, {8}};

  const nakshatra::ValueId x =
      g.add_op(nakshatra::OpKind::Input, {}, v, "x");
  const nakshatra::ValueId one =
      g.add_op(nakshatra::OpKind::Constant, {}, v, "one");
  set_int_attr(g, "fill", 1);
  const nakshatra::ValueId r =
      g.add_op(nakshatra::OpKind::Multiply, {x, one}, v);
  g.add_op(nakshatra::OpKind::Add, {r, x}, v);

  nakshatra::verify(g);

  const auto pass = nakshatra::create_canonicalize_pass();

  const bool first = pass->run(g);
  nakshatra::verify(g);
  const std::string once = g.dump();
  std::printf("\n--- idempotent, run pertama ---\n%s", once.c_str());

  const bool second = pass->run(g);
  nakshatra::verify(g);
  std::printf("--- run kedua ---\n%s", g.dump().c_str());

  check(first, "first canonicalize reports a change");
  check(!second, "second canonicalize is a no-op");
  check(g.dump() == once, "canonicalize is idempotent");

  const auto fuse = nakshatra::create_fuse_rmsnorm_linear_pass();
  check(!fuse->run(g), "fusion on an already-canonical graph is a no-op");
}

std::size_t num_elements(const nakshatra::TensorType &type) {
  std::size_t n = 1;
  for (std::size_t d : type.shape) {
    n *= d;
  }
  return n;
}

std::string vec_text(const std::vector<float> &v) {
  std::string s = "[";
  char buf[32];
  for (std::size_t i = 0; i < v.size(); ++i) {
    std::snprintf(buf, sizeof buf, "%s%.3f", i != 0 ? ", " : "", v[i]);
    s += buf;
  }
  return s + "]";
}

std::unordered_map<nakshatra::ValueId, std::vector<float>>
eval_graph(const nakshatra::Graph &g, const std::vector<float> &x_data) {
  std::unordered_map<nakshatra::ValueId, std::vector<float>> env;

  for (const nakshatra::Op &op : g.ops()) {
    switch (op.kind) {
    case nakshatra::OpKind::Input:
      env[op.result] = x_data;
      break;
    case nakshatra::OpKind::Constant: {
      const float fill = op.int_attrs.count("fill")
                             ? static_cast<float>(op.int_attrs.at("fill"))
                             : 0.0f;
      env[op.result].assign(num_elements(op.result_type), fill);
      break;
    }
    case nakshatra::OpKind::Multiply: {
      const auto &a = env.at(op.inputs[0]);
      const auto &b = env.at(op.inputs[1]);
      std::vector<float> y(a.size());
      for (std::size_t i = 0; i < a.size(); ++i) {
        y[i] = a[i] * b[i];
      }
      env[op.result] = std::move(y);
      break;
    }
    case nakshatra::OpKind::Add: {
      const auto &a = env.at(op.inputs[0]);
      const auto &b = env.at(op.inputs[1]);
      std::vector<float> y(a.size());
      for (std::size_t i = 0; i < a.size(); ++i) {
        y[i] = a[i] + b[i];
      }
      env[op.result] = std::move(y);
      break;
    }
    case nakshatra::OpKind::GELU: {
      const auto &a = env.at(op.inputs[0]);
      std::vector<float> y(a.size());
      for (std::size_t i = 0; i < a.size(); ++i) {
        y[i] = nakshatra::gelu_tanh(a[i]);
      }
      env[op.result] = std::move(y);
      break;
    }
    case nakshatra::OpKind::RMSNorm: {
      const auto &xv = env.at(op.inputs[0]);
      const auto &wv = env.at(op.inputs[1]);
      std::vector<float> y(xv.size());
      nakshatra::gemma_rmsnorm(xv, wv, 1.0e-6f, y);
      env[op.result] = std::move(y);
      break;
    }
    case nakshatra::OpKind::Linear:
    case nakshatra::OpKind::FusedRMSNormLinear: {
      const bool fused = op.kind == nakshatra::OpKind::FusedRMSNormLinear;
      std::vector<float> xv;
      if (fused) {
        const auto &x_in = env.at(op.inputs[0]);
        const auto &wv = env.at(op.inputs[1]);
        xv.resize(x_in.size());
        nakshatra::gemma_rmsnorm(x_in, wv, 1.0e-6f, xv);
      } else {
        xv = env.at(op.inputs[0]);
      }
      const auto &w = env.at(op.inputs[fused ? 2 : 1]);
      const std::size_t out = op.result_type.shape[0];
      const std::size_t in = xv.size();
      std::vector<float> y(out);
      nakshatra::linear(xv, w, out, in, y);
      env[op.result] = std::move(y);
      break;
    }
    default:
      break;
    }
  }

  return env;
}

bool all_close(const std::vector<float> &a, const std::vector<float> &b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::fabs(a[i] - b[i]) > 1.0e-5f) {
      return false;
    }
  }
  return true;
}

void test_execution_equivalence_numbers() {
  const std::vector<float> x{0.5f,  -1.0f, 2.0f, 0.25f,
                             -0.5f, 1.5f,  3.0f, -2.0f};
  const nakshatra::TensorType v{nakshatra::DType::F32, {8}};

  std::printf("\n=== ANGKA #1: x*1 ===\n");
  {
    nakshatra::Graph g;
    const nakshatra::ValueId xi =
        g.add_op(nakshatra::OpKind::Input, {}, v, "x");
    const nakshatra::ValueId one =
        g.add_op(nakshatra::OpKind::Constant, {}, v, "one");
    set_int_attr(g, "fill", 1);
    const nakshatra::ValueId r =
        g.add_op(nakshatra::OpKind::Multiply, {xi, one}, v);
    const nakshatra::ValueId a =
        g.add_op(nakshatra::OpKind::Add, {r, xi}, v);

    const auto before = eval_graph(g, x);
    std::printf("x  = %s\n", vec_text(before.at(xi)).c_str());
    std::printf("one= %s\n", vec_text(before.at(one)).c_str());
    std::printf("r  = x*1       = %s\n", vec_text(before.at(r)).c_str());
    std::printf("a  = r+x       = %s\n", vec_text(before.at(a)).c_str());

    run_pipeline(g, false, "multiply-one equivalence");

    const auto after = eval_graph(g, x);
    std::printf("a' = x+x       = %s\n", vec_text(after.at(a)).c_str());
    check(all_close(before.at(a), after.at(a)),
          "multiply-one: angka hasil identik");
  }

  std::printf("\n=== ANGKA #2: fusion rmsnorm+linear ===\n");
  {
    nakshatra::Graph g;
    const nakshatra::TensorType v16{nakshatra::DType::F32, {16, 8}};

    const nakshatra::ValueId xi =
        g.add_op(nakshatra::OpKind::Input, {}, v, "x");
    const nakshatra::ValueId w =
        g.add_op(nakshatra::OpKind::Constant, {}, v, "norm_w");
    set_int_attr(g, "fill", 0);
    const nakshatra::ValueId m =
        g.add_op(nakshatra::OpKind::Constant, {}, v16, "matrix");
    set_int_attr(g, "fill", 2);
    const nakshatra::ValueId n =
        g.add_op(nakshatra::OpKind::RMSNorm, {xi, w}, v);
    const nakshatra::ValueId y = g.add_op(
        nakshatra::OpKind::Linear, {n, m},
        nakshatra::TensorType{nakshatra::DType::F32, {16}});

    const auto before = eval_graph(g, x);
    std::printf("x  = %s\n", vec_text(before.at(xi)).c_str());
    std::printf("n  = rmsnorm(x)= %s\n", vec_text(before.at(n)).c_str());
    std::printf("y  = linear(n) = %s\n", vec_text(before.at(y)).c_str());

    run_pipeline(g, true, "fusion equivalence");

    const auto after = eval_graph(g, x);
    std::printf("y' = fused(x)  = %s\n", vec_text(after.at(y)).c_str());
    check(all_close(before.at(y), after.at(y)),
          "fusion: angka hasil identik");
  }

  std::printf("\n=== ANGKA #3: fusion DITOLAK (n dipake 2x) ===\n");
  {
    nakshatra::Graph g;
    const nakshatra::TensorType v16{nakshatra::DType::F32, {16, 8}};

    const nakshatra::ValueId xi =
        g.add_op(nakshatra::OpKind::Input, {}, v, "x");
    const nakshatra::ValueId w =
        g.add_op(nakshatra::OpKind::Constant, {}, v, "norm_w");
    set_int_attr(g, "fill", 0);
    const nakshatra::ValueId m =
        g.add_op(nakshatra::OpKind::Constant, {}, v16, "matrix");
    set_int_attr(g, "fill", 2);
    const nakshatra::ValueId n =
        g.add_op(nakshatra::OpKind::RMSNorm, {xi, w}, v);
    const nakshatra::ValueId y = g.add_op(
        nakshatra::OpKind::Linear, {n, m},
        nakshatra::TensorType{nakshatra::DType::F32, {16}});
    const nakshatra::ValueId h =
        g.add_op(nakshatra::OpKind::GELU, {n}, v);

    const auto before = eval_graph(g, x);
    std::printf("x  = %s\n", vec_text(before.at(xi)).c_str());
    std::printf("n  = rmsnorm(x)= %s   ← dipake DUA op\n",
                vec_text(before.at(n)).c_str());
    std::printf("y  = linear(n) = %s\n", vec_text(before.at(y)).c_str());
    std::printf("h  = gelu(n)   = %s\n", vec_text(before.at(h)).c_str());

    run_pipeline(g, true, "multi-use rejection equivalence");

    const auto after = eval_graph(g, x);
    std::printf("y' = %s\n", vec_text(after.at(y)).c_str());
    std::printf("h' = %s\n", vec_text(after.at(h)).c_str());
    check(all_close(before.at(y), after.at(y)) &&
              all_close(before.at(h), after.at(h)),
          "multi-use: angka hasil identik (fusion batal, gak ada yang rusak)");
  }
}

} // namespace

int main() {
  test_add_zero_redirects_every_use();
  test_multiply_one_redirects_every_use();
  test_rope_zero_folded_nonzero_preserved();
  test_fusion_rejects_multi_use_norm();
  test_fusion_happy_path();
  test_canonicalize_idempotent();
  test_execution_equivalence_numbers();

  if (g_failures != 0) {
    std::printf("%d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("all pass tests passed\n");
  return 0;
}

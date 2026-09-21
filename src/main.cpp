#include "config.hpp"
#include "exercises.hpp"
#include "kv_cache.hpp"
#include "metal_backend.hpp"
#include "model.hpp"
#include "ops.hpp"
#include "safetensors.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

struct PhaseTimes {
  double embed = 0;
  double layers = 0;
  double head = 0;
};

PhaseTimes profile_decode(const nakshatra::Gemma3Config &cfg,
                          const nakshatra::ModelWeights &model,
                          nakshatra::KVCache &cache, std::uint32_t token,
                          std::size_t position) {
  PhaseTimes t{};
  using clock = std::chrono::steady_clock;

  auto t0 = clock::now();
  std::vector<float> x(cfg.hidden_size);
  nakshatra::embed_token(cfg, model.embedding, token, x);
  auto t1 = clock::now();

  for (std::size_t layer = 0; layer < cfg.num_layers; ++layer) {
    nakshatra::decoder_layer_decode(cfg, model.layers[layer], cache, layer,
                                    position, x);
  }
  auto t2 = clock::now();

  std::vector<float> normed(cfg.hidden_size);
  nakshatra::gemma_rmsnorm(x, model.final_norm.span(), cfg.rms_eps, normed);

  std::vector<float> logits(cfg.vocab_size);
  nakshatra::compute_logits(cfg, model.embedding, normed, logits);
  auto t3 = clock::now();

  t.embed = std::chrono::duration<double>(t1 - t0).count();
  t.layers = std::chrono::duration<double>(t2 - t1).count();
  t.head = std::chrono::duration<double>(t3 - t2).count();
  return t;
}

int run_profile() {
  const std::filesystem::path path = "models/gemma-3-1b/model.safetensors";
  if (!std::filesystem::exists(path)) {
    std::cout << "model file not found\n";
    return 1;
  }

  std::cout << "loading...\n";
  nakshatra::Gemma3Config cfg;
  nakshatra::SafeTensorFile file = nakshatra::load_safetensors(path.string());
  nakshatra::ModelWeights model = nakshatra::load_gemma_weights(file, cfg);
  nakshatra::KVCache cache(cfg, 64);

  const std::vector<std::uint32_t> prompt{2, 9259, 1902};

  std::uint32_t next = 0;
  std::size_t pos = 0;
  for (; pos < prompt.size(); ++pos) {
    next = nakshatra::decode_one(cfg, model, cache, prompt[pos], pos);
  }

  PhaseTimes total{};
  const int steps = 5;
  std::cout << "profiling " << steps << " tokens...\n";

  for (int i = 0; i < steps; ++i, ++pos) {
    const PhaseTimes t = profile_decode(cfg, model, cache, next, pos);
    total.embed += t.embed;
    total.layers += t.layers;
    total.head += t.head;
  }

  const double all = total.embed + total.layers + total.head;
  std::cout << "embed        : " << total.embed << "s ("
            << 100.0 * total.embed / all << "%)\n";
  std::cout << "26 layers    : " << total.layers << "s ("
            << 100.0 * total.layers / all << "%)\n";
  std::cout << "norm+logits  : " << total.head << "s ("
            << 100.0 * total.head / all << "%)\n";
  std::cout << "total        : " << all << "s (" << all / steps * 1000.0
            << " ms/token)\n";
  return 0;
}

int run_app(bool metal_on, nakshatra::WeightFormat fmt) {
  const std::filesystem::path path = "models/gemma-3-1b/model.safetensors";

  if (!std::filesystem::exists(path)) {
    std::cout << "model file not found: " << path << "\n";
    std::cout << "download it first, or run './nakshatra --exercises'\n";
    return 1;
  }

  nakshatra::global_metal().set_enabled(metal_on);
  nakshatra::global_metal().set_weight_format(fmt);

  const bool using_metal = nakshatra::global_metal().ok() && metal_on;

  std::cout << "backend: "
            << (using_metal
                    ? (fmt == nakshatra::WeightFormat::BF16
                           ? "metal, bf16 weights (2.4 GB gpu)"
                           : "metal, f32 weights (4.8 GB gpu)")
                    : "cpu")
            << "\n";
  std::cout << "loading gemma 3 1b (2 GB)...\n";
  const auto t0 = std::chrono::steady_clock::now();

  nakshatra::Gemma3Config cfg;
  nakshatra::ModelWeights model;
  {
    nakshatra::SafeTensorFile file = nakshatra::load_safetensors(path.string());
    model = nakshatra::load_gemma_weights(file, cfg);
  }

  const auto t1 = std::chrono::steady_clock::now();
  std::cout << "weights loaded in "
            << std::chrono::duration<double>(t1 - t0).count() << "s\n";

  nakshatra::KVCache cache(cfg, 64);

  const std::vector<std::uint32_t> prompt{2, 9259, 1902};

  std::uint32_t next = 0;
  std::size_t pos = 0;

  for (; pos < prompt.size(); ++pos) {
    next = nakshatra::decode_one(cfg, model, cache, prompt[pos], pos);
  }

  std::cout << "generated token ids:";
  const auto t2 = std::chrono::steady_clock::now();

  for (std::size_t step = 0; step < 12; ++step, ++pos) {
    next = nakshatra::decode_one(cfg, model, cache, next, pos);
    std::cout << " " << next << std::flush;
  }

  const auto t3 = std::chrono::steady_clock::now();
  const double gen_s = std::chrono::duration<double>(t3 - t2).count();

  std::cout << "\n12 tokens in " << gen_s << "s ("
            << gen_s / 12.0 * 1000.0 << " ms/token, " << 12.0 / gen_s
            << " tok/s)\n";

  return 0;
}

} // namespace

int run_metal() {
  nakshatra::MetalMatvec metal;

  if (!metal.ok()) {
    std::cout << "metal device not available\n";
    return 1;
  }

  std::cout << "metal matvec benchmark (naive kernel, fp32)\n";
  std::cout << "shape             | cpu ms | metal ms | speedup | metal GB/s | "
               "GFLOP/s\n";
  std::cout << "-------------------------------------------------------------"
               "-------------\n";

  const std::vector<std::pair<std::size_t, std::size_t>> shapes{
      {1024, 1152}, {6912, 1152}, {1152, 6912}, {262144, 1152}};

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-0.02f, 0.02f);

  for (const auto &[out, in] : shapes) {
    std::vector<float> w(out * in);
    for (float &v : w) {
      v = dist(gen);
    }
    std::vector<float> x(in);
    for (float &v : x) {
      v = dist(gen);
    }

    std::vector<float> y_cpu(out, 0.0f);
    std::vector<float> y_metal(out, 0.0f);

    if (!metal.run(w.data(), x.data(), y_metal.data(), out, in)) {
      std::cout << "metal run failed for shape " << out << "x" << in << "\n";
      return 1;
    }

    nakshatra::linear(x, w, out, in, y_cpu);

    double max_diff = 0.0;
    for (std::size_t i = 0; i < out; ++i) {
      max_diff = std::max(max_diff,
                          std::abs(static_cast<double>(y_cpu[i]) -
                                   static_cast<double>(y_metal[i])));
    }

    const std::size_t cpu_iters =
        std::max<std::size_t>(3, std::min<std::size_t>(100, 400000000ull /
                                                                  (out * in)));
    const std::size_t gpu_iters =
        std::max<std::size_t>(20, std::min<std::size_t>(400, 4000000000ull /
                                                                   (out * in)));

    const auto c0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < cpu_iters; ++i) {
      nakshatra::linear(x, w, out, in, y_cpu);
    }
    const auto c1 = std::chrono::steady_clock::now();

    const double cpu_s =
        std::chrono::duration<double>(c1 - c0).count() / cpu_iters;
    const double gpu_s = metal.bench(w.data(), x.data(), y_metal.data(), out,
                                     in, 10, gpu_iters);

    const double bytes =
        static_cast<double>(out * in + in + out) * 4.0;
    const double gbs = bytes / gpu_s / 1e9;
    const double gflo = 2.0 * out * in / gpu_s / 1e9;

    std::cout << std::left << std::setw(17) << (std::to_string(out) + "x" +
                                                std::to_string(in))
              << " | " << std::setw(6) << cpu_s * 1000.0 << " | "
              << std::setw(8) << gpu_s * 1000.0 << " | " << std::setw(7)
              << std::fixed << std::setprecision(1) << cpu_s / gpu_s << " | "
              << std::setw(10) << gbs << " | " << std::setw(7) << gflo
              << "  max diff " << max_diff << "\n";
  }

  return 0;
}

int main(int argc, char **argv) {
  const std::string arg = argc > 1 ? argv[1] : "";

  if (arg == "--exercises") {
    nakshatra::run_all_exercises();
    return 0;
  }

  if (arg == "--profile") {
    return run_profile();
  }

  if (arg == "--metal") {
    return run_metal();
  }

  if (arg == "--cpu") {
    return run_app(false, nakshatra::WeightFormat::F32);
  }

  if (arg == "--bf16") {
    return run_app(true, nakshatra::WeightFormat::BF16);
  }

  return run_app(true, nakshatra::WeightFormat::F32);
}

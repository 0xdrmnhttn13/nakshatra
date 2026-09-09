#include "pass.hpp"

#include <algorithm>
#include <cstdint>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace nakshatra {
namespace {

const Op *find_def(const Graph &graph, ValueId value) {
  for (const Op &op : graph.ops()) {
    if (op.result == value) {
      return &op;
    }
  }
  return nullptr;
}

bool is_fill_constant(const Graph &graph, ValueId value, std::int64_t fill) {
  const Op *op = find_def(graph, value);
  if (op == nullptr || op->kind != OpKind::Constant) {
    return false;
  }
  const auto it = op->int_attrs.find("fill");
  return it != op->int_attrs.end() && it->second == fill;
}

std::size_t eliminate_dead_ops(Graph &graph,
                                const std::unordered_set<ValueId> &removable) {
  std::size_t removed_total = 0;

  bool removed_any = true;
  while (removed_any) {
    removed_any = false;

    std::unordered_set<ValueId> used;
    for (const Op &op : graph.ops()) {
      used.insert(op.inputs.begin(), op.inputs.end());
    }

    auto &ops = graph.ops();
    const auto dead = std::remove_if(
        ops.begin(), ops.end(), [&used, &removable](const Op &op) {
          return op.kind != OpKind::Input && !used.contains(op.result) &&
                 (removable.contains(op.result) ||
                  op.kind == OpKind::Constant);
        });

    if (dead != ops.end()) {
      removed_total += static_cast<std::size_t>(ops.end() - dead);
      ops.erase(dead, ops.end());
      removed_any = true;
    }
  }

  return removed_total;
}

class CanonicalizePass final : public Pass {
public:
  std::string_view name() const override { return "canonicalize"; }

  bool run(Graph &graph) override {
    struct Rewrite {
      ValueId from;
      ValueId to;
    };

    std::vector<Rewrite> rewrites;

    for (const Op &op : graph.ops()) {
      if ((op.kind == OpKind::Add || op.kind == OpKind::Multiply) &&
          op.inputs.size() == 2) {
        const std::int64_t identity = op.kind == OpKind::Add ? 0 : 1;
        for (std::size_t i = 0; i < 2; ++i) {
          const Op *other = find_def(graph, op.inputs[1 - i]);
          if (is_fill_constant(graph, op.inputs[i], identity) &&
              other != nullptr &&
              other->result_type == op.result_type) {
            rewrites.push_back({op.result, op.inputs[1 - i]});
            break;
          }
        }
      } else if (op.kind == OpKind::RoPE && op.inputs.size() == 1) {
        const auto position = op.int_attrs.find("position");
        const Op *input = find_def(graph, op.inputs[0]);
        if (position != op.int_attrs.end() && position->second == 0 &&
            input != nullptr && input->result_type == op.result_type) {
          rewrites.push_back({op.result, op.inputs[0]});
        }
      }
    }

    std::unordered_set<ValueId> victims;
    for (const Rewrite &rewrite : rewrites) {
      graph.replace_all_uses(rewrite.from, rewrite.to);
      victims.insert(rewrite.from);
    }

    const std::size_t removed = eliminate_dead_ops(graph, victims);
    return !rewrites.empty() || removed > 0;
  }
};

bool fused_rmsnorm_linear_legal(const Graph &graph, const Op &norm,
                                const Op &linear) {
  if (norm.result_type.dtype != DType::F32 ||
      linear.result_type.dtype != DType::F32) {
    return false;
  }
  const Op *x = find_def(graph, norm.inputs[0]);
  const Op *matrix = find_def(graph, linear.inputs[1]);
  return x != nullptr && matrix != nullptr &&
         x->result_type.shape.size() == 1 &&
         matrix->result_type.shape.size() == 2;
}

class FuseRMSNormLinearPass final : public Pass {
public:
  std::string_view name() const override { return "fuse-rmsnorm-linear"; }

  bool run(Graph &graph) override {
    bool changed = false;
    std::unordered_set<ValueId> victims;

    for (Op &op : graph.ops()) {
      if (op.kind != OpKind::Linear || op.inputs.size() != 2) {
        continue;
      }

      const Op *norm = find_def(graph, op.inputs[0]);
      if (norm == nullptr || norm->kind != OpKind::RMSNorm ||
          norm->inputs.size() != 2) {
        continue;
      }

      if (graph.use_count(op.inputs[0]) != 1) {
        continue;
      }

      if (!fused_rmsnorm_linear_legal(graph, *norm, op)) {
        continue;
      }

      const ValueId x = norm->inputs[0];
      const ValueId w = norm->inputs[1];
      const ValueId matrix = op.inputs[1];
      const ValueId norm_result = norm->result;

      op.kind = OpKind::FusedRMSNormLinear;
      op.inputs = {x, w, matrix};
      victims.insert(norm_result);
      changed = true;
    }

    if (changed) {
      eliminate_dead_ops(graph, victims);
    }

    return changed;
  }
};

} // namespace

std::unique_ptr<Pass> create_canonicalize_pass() {
  return std::make_unique<CanonicalizePass>();
}

std::unique_ptr<Pass> create_fuse_rmsnorm_linear_pass() {
  return std::make_unique<FuseRMSNormLinearPass>();
}

} // namespace nakshatra

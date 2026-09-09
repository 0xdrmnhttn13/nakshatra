#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace nakshatra {
enum class DType { F32, BF16 };
struct TensorType {
  DType dtype = DType::F32;
  std::vector<std::size_t> shape;
  bool operator==(const TensorType &) const = default;
};

using ValueId = std::uint32_t;

enum class OpKind {
  Input,
  Constant,
  Linear,
  RMSNorm,
  RoPE,
  Attention,
  GELU,
  Multiply,
  Add,
  FusedRMSNormLinear
};

struct Op {
  OpKind kind;
  std::vector<ValueId> inputs;
  ValueId result;
  TensorType result_type;
  std::unordered_map<std::string, std::int64_t> int_attrs;
  std::string debug_name;
};

class Graph {
public:
  ValueId add_op(OpKind kind, std::vector<ValueId> inputs,
                 TensorType result_type, std::string debug_name = {});
  const std::vector<Op> &ops() const { return ops_; }
  std::vector<Op> &ops() { return ops_; }
  const Op &defining_op(ValueId value) const;
  std::size_t replace_all_uses(ValueId old_value, ValueId new_value);
  std::size_t use_count(ValueId value) const;
  std::string dump() const;

private:
  std::vector<Op> ops_;
  ValueId next_value_ = 0;
};

void verify(const Graph &graph);
} // namespace nakshatra

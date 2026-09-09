#include "ir.hpp"

#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace nakshatra {

static std::string kind_name(OpKind kind) {
  switch (kind) {
  case OpKind::Input:
    return "input";
  case OpKind::Constant:
    return "constant";
  case OpKind::Linear:
    return "linear";
  case OpKind::RMSNorm:
    return "rmsnorm";
  case OpKind::RoPE:
    return "rope";
  case OpKind::Attention:
    return "attention";
  case OpKind::GELU:
    return "gelu";
  case OpKind::Multiply:
    return "multiply";
  case OpKind::Add:
    return "add";
  case OpKind::FusedRMSNormLinear:
    return "fused_rmsnorm_linear";
  }
  return "unknown";
}

static std::string type_text(const TensorType &type) {
  std::string text = type.dtype == DType::F32 ? "f32" : "bf16";
  text += "[";
  for (std::size_t i = 0; i < type.shape.size(); ++i) {
    if (i != 0) {
      text += ",";
    }
    text += std::to_string(type.shape[i]);
  }
  text += "]";
  return text;
}

std::string Graph::dump() const {
  std::ostringstream out;

  for (const Op &op : ops_) {
    out << "%" << op.result << " = " << kind_name(op.kind);

    for (ValueId input : op.inputs) {
      out << " %" << input;
    }

    if (!op.debug_name.empty()) {
      out << " (" << op.debug_name << ")";
    }

    out << " : " << type_text(op.result_type) << "\n";
  }

  return out.str();
}

ValueId Graph::add_op(OpKind kind, std::vector<ValueId> inputs,
                      TensorType result_type, std::string debug_name) {
  const ValueId result = next_value_++;
  ops_.push_back(Op{kind,
                    std::move(inputs),
                    result,
                    std::move(result_type),
                    {},
                    std::move(debug_name)});
  return result;
}

const Op &Graph::defining_op(ValueId value) const {
  for (const Op &op : ops_) {
    if (op.result == value) {
      return op;
    }
  }
  throw std::runtime_error("value has no defining operation");
}

std::size_t Graph::replace_all_uses(ValueId old_value, ValueId new_value) {
  std::size_t replaced = 0;
  for (Op &op : ops_) {
    for (ValueId &input : op.inputs) {
      if (input == old_value) {
        input = new_value;
        ++replaced;
      }
    }
  }
  return replaced;
}

std::size_t Graph::use_count(ValueId value) const {
  std::size_t count = 0;
  for (const Op &op : ops_) {
    for (ValueId input : op.inputs) {
      if (input == value) {
        ++count;
      }
    }
  }
  return count;
}

void verify(const Graph &graph) {
  std::unordered_set<ValueId> defined;

  for (const Op &op : graph.ops()) {
    for (ValueId input : op.inputs) {
      if (!defined.contains(input)) {
        throw std::runtime_error(op.debug_name + ": use before definition");
      }
    }

    if (op.result_type.shape.empty()) {
      throw std::runtime_error(op.debug_name +
                               ": scalar/rank-0 not supported yet");
    }

    if (!defined.insert(op.result).second) {
      throw std::runtime_error(op.debug_name + ": duplicate SSA value");
    }
  }
}

} // namespace nakshatra

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.hpp"

namespace nakshatra {

struct TensorMeta {
  std::string dtype;
  std::vector<std::size_t> shape;
  std::size_t begin = 0;
  std::size_t end = 0;
};

struct SafeTensorFile {
  std::vector<std::byte> bytes;
  std::size_t data_start = 0;
  std::unordered_map<std::string, TensorMeta> tensors;
};

SafeTensorFile load_safetensors(const std::string &path);

Tensor materialize_bf16(const SafeTensorFile &file, const std::string &name);

} // namespace nakshatra

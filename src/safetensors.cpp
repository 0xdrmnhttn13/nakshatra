#include "safetensors.hpp"

#include <fstream>
#include <iterator>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace nakshatra {

std::uint64_t read_u64_le(const std::byte *p) {
  std::uint64_t value = 0;

  for (int i = 0; i < 8; ++i) {
    value |= static_cast<std::uint64_t>(
                 std::to_integer<unsigned char>(p[i]))
             << (8 * i);
  }

  return value;
}

std::vector<std::byte> read_all_bytes(const std::string &path) {
  std::ifstream in(path, std::ios::binary);

  if (!in) {
    throw std::runtime_error("cannot open " + path);
  }

  in.seekg(0, std::ios::end);
  const auto size = in.tellg();
  in.seekg(0, std::ios::beg);

  std::vector<std::byte> bytes(static_cast<std::size_t>(size));

  in.read(reinterpret_cast<char *>(bytes.data()), size);

  return bytes;
}

SafeTensorFile load_safetensors(const std::string &path) {
  SafeTensorFile file;
  file.bytes = read_all_bytes(path);

  if (file.bytes.size() < 8) {
    throw std::runtime_error("invalid safetensors file");
  }

  const auto header_len = read_u64_le(file.bytes.data());

  const std::size_t header_begin = 8;
  const std::size_t header_end =
      header_begin + static_cast<std::size_t>(header_len);

  if (header_end > file.bytes.size()) {
    throw std::runtime_error("invalid safetensors header length");
  }

  std::string json_text(
      reinterpret_cast<const char *>(file.bytes.data() + header_begin),
      static_cast<std::size_t>(header_len));

  const auto json = nlohmann::json::parse(json_text);

  file.data_start = header_end;

  for (auto it = json.begin(); it != json.end(); ++it) {
    if (it.key() == "__metadata__") {
      continue;
    }

    TensorMeta meta;

    meta.dtype = it.value().at("dtype").get<std::string>();

    meta.shape =
        it.value().at("shape").get<std::vector<std::size_t>>();

    auto offsets =
        it.value().at("data_offsets").get<std::vector<std::size_t>>();

    meta.begin = offsets.at(0);
    meta.end = offsets.at(1);

    file.tensors.emplace(it.key(), std::move(meta));
  }

  return file;
}

Tensor materialize_bf16(const SafeTensorFile &file,
                        const std::string &name) {
  const auto &meta = file.tensors.at(name);

  if (meta.dtype != "BF16") {
    throw std::runtime_error("expected BF16 for " + name);
  }

  const std::size_t byte_count = meta.end - meta.begin;

  if (byte_count % 2 != 0) {
    throw std::runtime_error("odd BF16 byte count");
  }

  const std::byte *raw =
      file.bytes.data() + file.data_start + meta.begin;

  const std::size_t n = byte_count / 2;

  std::vector<float> out(n);

  for (std::size_t i = 0; i < n; ++i) {
    const auto lo = std::to_integer<std::uint8_t>(raw[2 * i]);
    const auto hi = std::to_integer<std::uint8_t>(raw[2 * i + 1]);

    const std::uint16_t b =
        static_cast<std::uint16_t>(lo) |
        (static_cast<std::uint16_t>(hi) << 8);

    out[i] = bf16_to_float(b);
  }

  return Tensor(meta.shape, std::move(out));
}

} // namespace nakshatra

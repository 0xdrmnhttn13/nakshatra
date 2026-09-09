#pragma once

#include <cassert>
#include <cstddef>
#include <numeric>
#include <span>

#include <stdexcept>
#include <vector>

#include <bit>
#include <cstdint>

namespace nakshatra {
class Tensor {
public:
  Tensor() = default;
  explicit Tensor(std::vector<std::size_t> shape)
      : shape_(std::move(shape)), data_(numel_from(shape_), 0.0f) {}
  Tensor(std::vector<std::size_t> shape, std::vector<float> data)
      : shape_(std::move(shape)), data_(std::move(data)) {
    if (data_.size() != numel_from(shape_)) {
      throw std::runtime_error("tensor shape/data mismatch");
    }
  }
  std::size_t rank() const { return shape_.size(); }
  std::size_t numel() const { return data_.size(); }

  std::size_t dim(std::size_t i) const { return shape_.at(i); }

  const std::vector<std::size_t> &shape() const { return shape_; }

  float *data() { return data_.data(); }

  const float *data() const { return data_.data(); }

  std::span<float> span() { return data_; }
  std::span<const float> span() const { return data_; }

  float &operator[](std::size_t i) { return data_.at(i); }
  float operator[](std::size_t i) const { return data_.at(i); }

private:
  static std::size_t numel_from(const std::vector<std::size_t> &shape) {
    std::size_t n = 1;
    for (auto d : shape) {
      n *= d;
    }
    return n;
  }
  std::vector<std::size_t> shape_;
  std::vector<float> data_;
};

inline float bf16_to_float(std::uint16_t x) {
  std::uint32_t bits = static_cast<std::uint32_t>(x) << 16;

  return std::bit_cast<float>(bits);
}

inline std::uint16_t float_to_bf16_trunc(float x) {
  std::uint32_t bits = std::bit_cast<std::uint32_t>(x);
  return static_cast<std::uint16_t>(bits >> 16);
}
} // namespace nakshatra

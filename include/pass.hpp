#pragma once

#include "ir.hpp"

#include <memory>
#include <string_view>
#include <vector>

namespace nakshatra {
class Pass {
public:
  virtual ~Pass() = default;
  virtual std::string_view name() const = 0;
  virtual bool run(Graph &graph) = 0;
};

class PassManager {
public:
  void add(std::unique_ptr<Pass> pass);
  void run(Graph &graph, bool verify_each = true);

private:
  std::vector<std::unique_ptr<Pass>> passes_;
};

std::unique_ptr<Pass> create_canonicalize_pass();
std::unique_ptr<Pass> create_fuse_rmsnorm_linear_pass();
} // namespace nakshatra

#include "pass.hpp"
#include <stdexcept>

namespace nakshatra {
void PassManager::add(std::unique_ptr<Pass> pass) {
  passes_.push_back(std::move(pass));
}
void PassManager::run(Graph &graph, bool verify_each) {
  if (verify_each) {
    verify(graph);
  }
  for (auto &pass : passes_) {
    pass->run(graph);
    if (verify_each) {
      try {
        verify(graph);
      } catch (const std::exception &error) {
        throw std::runtime_error(std::string(pass->name()) +
                                 " produce invalid IR: " + error.what());
      }
    }
  }
}
} // namespace nakshatra

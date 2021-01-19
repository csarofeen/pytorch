
#include <torch/csrc/jit/passes/automatic_mixed_precision.h>

#include <torch/csrc/jit/jit_log.h>

namespace torch {
namespace jit {

void AutomaticMixedPrecision(const std::shared_ptr<Graph>& graph) {
  // TODO
  GRAPH_DUMP("After AutomaticMixedPrecision: ", graph);
}

} // namespace jit
} // namespace torch

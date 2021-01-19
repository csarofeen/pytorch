
#include <torch/csrc/jit/passes/automatic_mixed_precision.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>

namespace torch {
namespace jit {

namespace {

void handleBlock(Block* block) {
  for (Node* node : block->nodes()) {
    for (Block* sub_block : node->blocks()) {
      handleBlock(sub_block);
    }

    // TODO
  }
}

} // namespace

void AutomaticMixedPrecision(const std::shared_ptr<Graph>& graph) {
  GRAPH_DUMP("Before AutomaticMixedPrecision: ", graph);
  // TODO
  GRAPH_DUMP("After AutomaticMixedPrecision: ", graph);
}

} // namespace jit
} // namespace torch

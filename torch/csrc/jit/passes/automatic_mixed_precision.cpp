
#include <torch/csrc/jit/passes/automatic_mixed_precision.h>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>

namespace torch {
namespace jit {

namespace {

void handleBlock(Block* block, bool initial_autocast_enabled) {
  bool autocast_enabled = initial_autocast_enabled;

  for (Node* node : block->nodes()) {
    switch (node->kind()) {
      case prim::CallFunction:
      case prim::CallMethod:
        TORCH_INTERNAL_ASSERT(false, "Calls are not supported with AMP & JIT");
        break;
      case prim::CreateObject:
        break;
      case prim::SetAttr:
        break;
      case prim::Enter:
        break;
      case prim::Exit:
        break;
    }

    // process sub-blocks, if any
    for (Block* sub_block : node->blocks()) {
      handleBlock(sub_block, autocast_enabled);
    }
  }

  // Sanity check: make sure there's no unbalanced transition
  TORCH_INTERNAL_ASSERT(autocast_enabled == initial_autocast_enabled);
}

} // namespace

void AutomaticMixedPrecision(const std::shared_ptr<Graph>& graph) {
  GRAPH_DUMP("Before AutomaticMixedPrecision: ", graph);
  handleBlock(graph->block(), false);
  GRAPH_DUMP("After AutomaticMixedPrecision: ", graph);
}

} // namespace jit
} // namespace torch

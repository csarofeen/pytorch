
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
        // TODO
        break;

      case prim::SetAttr:
        // TODO
        break;

      case prim::Enter:
        // TODO
        break;

      case prim::Exit:
        // TODO
        break;

      // CastPolicy::fp16 (cast all inputs to float16)
      case aten::_convolution:
      case aten::_convolution_nogroup:
      case aten::conv1d:
      case aten::conv2d:
      case aten::conv3d:
      case aten::conv_tbc:
      case aten::conv_transpose1d:
      case aten::convolution:
      case aten::cudnn_convolution:
      case aten::cudnn_convolution_transpose:
      case aten::prelu:
      case aten::addmm:
      case aten::addmv:
      case aten::addr:
      case aten::matmul:
      case aten::mm:
      case aten::mv:
      case aten::linear:
      case aten::addbmm:
      case aten::baddbmm:
      case aten::bmm:
      case aten::chain_matmul:
      case aten::_thnn_fused_lstm_cell:
      case aten::_thnn_fused_gru_cell:
      case aten::lstm_cell:
      case aten::gru_cell:
      case aten::rnn_tanh_cell:
      case aten::rnn_relu_cell:
        // TODO
        break;

      // CastPolicy::fp32 (cast all inputs to float32)
      case aten::native_layer_norm:
      case aten::acos:
      case aten::asin:
      case aten::cosh:
      case aten::erfinv:
      case aten::exp:
      case aten::expm1:
      case aten::log:
      case aten::log10:
      case aten::log2:
      case aten::log1p:
      case aten::reciprocal:
      case aten::rsqrt:
      case aten::sinh:
      case aten::tan:
      case aten::pow:
      case aten::softplus:
      case aten::gelu:
      case aten::layer_norm:
      case aten::group_norm:
      case aten::frobenius_norm:
      case aten::nuclear_norm:
      case aten::cosine_similarity:
      case aten::cosine_embedding_loss:
      case aten::nll_loss:
      case aten::nll_loss2d:
      case aten::hinge_embedding_loss:
      case aten::kl_div:
      case aten::l1_loss:
      case aten::smooth_l1_loss:
      case aten::mse_loss:
      case aten::margin_ranking_loss:
      case aten::multilabel_margin_loss:
      case aten::soft_margin_loss:
      case aten::triplet_margin_loss:
      case aten::multi_margin_loss:
      case aten::binary_cross_entropy_with_logits:
      case aten::dist:
      case aten::pdist:
      case aten::cdist:
      case aten::renorm:
        // TODO
        break;

      // CastPolicy::promote (promote inputs to the widest type)
      case aten::addcdiv:
      case aten::addcmul:
      case aten::atan2:
      case aten::bilinear:
      case aten::cat:
      case aten::_cat:
      case aten::cross:
      case aten::dot:
      case aten::equal:
      case aten::index_put:
      case aten::stack:
      case aten::tensordot:
        // TODO
        break;

      // Banned in autocast, see binary_cross_entropy_banned()
      case aten::binary_cross_entropy:
        AT_ERROR("Unsafe to autocast");
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

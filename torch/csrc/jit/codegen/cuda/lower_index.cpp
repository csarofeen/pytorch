
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/predicate_compute.h>

#include <torch/csrc/jit/codegen/cuda/lower_index.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

IndexLowering::IndexLowering() : ir_builder_(GpuLower::current()->kernel()) {}

kir::Val* IndexLowering::lowerSrcIndex(kir::Val* val, kir::Val* dst) const {
  if (auto tv = dynamic_cast<kir::TensorView*>(val)) {
    TORCH_INTERNAL_ASSERT(dst->isA<kir::TensorView>());
    return Index::getProducerIndex(
        tv->fuserTv(),
        dst->as<kir::TensorView>()->fuserTv(),
        scope_utils::getLoops(active_scope_expr_));
  } else {
    return val;
  }
}

kir::Val* IndexLowering::lowerDstIndex(kir::Val* dst) const {
  if (auto tv = dynamic_cast<kir::TensorView*>(dst)) {
    return Index::getConsumerIndex(
        tv->fuserTv(), scope_utils::getLoops(active_scope_expr_));
  } else {
    return dst;
  }
}

void IndexLowering::pushBack(kir::Expr* expr) {
  if (active_scope_ == nullptr) {
    lowered_exprs_.push_back(expr);
  } else {
    active_scope_->push_back(expr);
  }
}

void IndexLowering::visit(const kir::IfThenElse* ite) {
  const auto prev_scope_expr = active_scope_expr_;
  const auto prev_scope = active_scope_;

  // TODO(kir): try to avoid recreating new nodes and leaving old ones around
  auto new_ite =
      ir_builder_.create<kir::IfThenElse>(ite->cond(), prev_scope_expr);
  pushBack(new_ite);

  active_scope_expr_ = new_ite;
  active_scope_ = &new_ite->thenBody();

  for (auto expr : ite->thenBody().exprs()) {
    expr->accept(this);
  }

  active_scope_ = &new_ite->elseBody();

  for (auto expr : ite->elseBody().exprs()) {
    expr->accept(this);
  }

  active_scope_ = prev_scope;
  active_scope_expr_ = prev_scope_expr;
}

void IndexLowering::visit(const kir::ForLoop* for_loop) {
  const auto prev_scope_expr = active_scope_expr_;
  const auto prev_scope = active_scope_;

  auto new_for_loop = ir_builder_.create<kir::ForLoop>(
      for_loop->index(), for_loop->iter_domain(), prev_scope_expr);
  pushBack(new_for_loop);

  active_scope_expr_ = new_for_loop;
  active_scope_ = &new_for_loop->body();

  for (auto expr : for_loop->body().exprs()) {
    expr->accept(this);
  }

  active_scope_ = prev_scope;
  active_scope_expr_ = prev_scope_expr;
}

void IndexLowering::visit(const kir::UnaryOp* uop) {
  const auto in = lowerSrcIndex(uop->in(), uop->out());
  const auto out = lowerDstIndex(uop->out());
  pushBack(ir_builder_.create<kir::UnaryOp>(uop->operation(), out, in));
}

void IndexLowering::visit(const kir::BinaryOp* bop) {
  const auto lhs = lowerSrcIndex(bop->lhs(), bop->out());
  const auto rhs = lowerSrcIndex(bop->rhs(), bop->out());
  const auto out = lowerDstIndex(bop->out());
  pushBack(ir_builder_.create<kir::BinaryOp>(bop->operation(), out, lhs, rhs));
}

void IndexLowering::visit(const kir::TernaryOp* top) {
  const auto in1 = lowerSrcIndex(top->in1(), top->out());
  const auto in2 = lowerSrcIndex(top->in2(), top->out());
  const auto in3 = lowerSrcIndex(top->in3(), top->out());
  const auto out = lowerDstIndex(top->out());
  pushBack(
      ir_builder_.create<kir::TernaryOp>(top->operation(), out, in1, in2, in3));
}

namespace {

void allocateGridReductionFlag(
    kir::TensorView* out_tv,
    kir::Expr* current_scope_expr) {
  kir::IrBuilder ir_builder(GpuLower::current()->kernel());

  const auto flag_name = kir::GridReduction::getPredicateFlagName(out_tv);
  const auto flag_var = ir_builder.create<kir::Allocate>(
      ir_builder.create<kir::NamedScalar>(flag_name, DataType::Bool),
      MemoryType::Local,
      ir_builder.create<kir::Int>(1));

  // When enclosed by IfThenElse, place the variable outside of the
  // IfThenElse. This IfThenElse is assumed to be the prediate for
  // this grid reduction expression.
  if (current_scope_expr->isA<kir::IfThenElse>()) {
    scope_utils::insertBefore(
        current_scope_expr->parentScope(), current_scope_expr, flag_var);
  } else {
    TORCH_INTERNAL_ASSERT(current_scope_expr->isA<kir::ForLoop>());
    current_scope_expr->as<kir::ForLoop>()->body().push_back(flag_var);
  }
}

} // namespace

void IndexLowering::visit(const kir::ReductionOp* rop) {
  TORCH_INTERNAL_ASSERT(ir_utils::isTVOp(rop));

  const auto gpu_lower = GpuLower::current();

  const auto out_tv = rop->out()->as<kir::TensorView>();
  const auto out_domain = out_tv->domain();

  const bool is_block_reduce = out_domain->hasBlockReduction();
  const bool is_grid_reduce = out_domain->hasGridReduction();

  // If we do a grid reduction we can't have a reduction axis that is not bound
  // to a grid or block dim ()
  if (is_grid_reduce) {
    TORCH_INTERNAL_ASSERT(
        std::none_of(
            out_domain->domain().begin(),
            out_domain->domain().end(),
            [](kir::IterDomain* id) {
              return !id->isThread() && id->isReduction();
            }),
        "Found a reduction stage that has both a non-parallelized ",
        "reduction and a grid reduction.  This is not supported, ",
        "please use rfactor to do the serialized reduction first, ",
        "then the grid reduction.");
  }

  const auto out = lowerDstIndex(rop->out());
  const auto in = lowerSrcIndex(rop->in(), rop->out());

  const auto pred = PredicateCompute::getInlinePredicate(
      rop, scope_utils::getLoops(active_scope_expr_), nullptr, false);

  kir::ReductionOp* block_reduction_op = nullptr;

  if (is_block_reduce) {
    block_reduction_op = ir_builder_.create<kir::ReductionOp>(
        rop->operation(), rop->init(), out, in);
    block_reduction_op->setPredicate(pred);
    pushBack(block_reduction_op);
  }

  if (is_grid_reduce) {
    // First, declare a boolean flag variable storing the return value
    // of the gridReduce() helper
    allocateGridReductionFlag(out_tv, active_scope_expr_);

    auto buffer_ids = out_domain->domain();
    buffer_ids.erase(
        std::remove_if(
            buffer_ids.begin(),
            buffer_ids.end(),
            [](kir::IterDomain* id) {
              return id->isReduction() && !id->isBlockDim();
            }),
        buffer_ids.end());

    kir::Val* buffer_size = buffer_ids.empty() ? ir_builder_.create<kir::Int>(1)
                                               : buffer_ids[0]->rawExtent();

    for (size_t i = 1; i < buffer_ids.size(); i++) {
      buffer_size =
          ir_builder_.mulExpr(buffer_size, buffer_ids[i]->rawExtent());
    }

    auto sync_ids = out_domain->domain();
    sync_ids.erase(
        std::remove_if(
            sync_ids.begin(),
            sync_ids.end(),
            [](kir::IterDomain* id) {
              return id->isReduction() || !id->isBlockDim();
            }),
        sync_ids.end());

    kir::Val* sync_size = sync_ids.empty() ? ir_builder_.create<kir::Int>(1)
                                           : sync_ids[0]->rawExtent();

    for (size_t i = 1; i < sync_ids.size(); i++) {
      sync_size = ir_builder_.mulExpr(sync_size, sync_ids[i]->rawExtent());
    }

    const auto zero = ir_builder_.create<kir::Int>(0);

    const std::vector<kir::IterDomain*> new_buffer_ids = {
        ir_builder_.create<kir::IterDomain>(zero, buffer_size)};
    const auto buffer_domain =
        ir_builder_.create<kir::TensorDomain>(new_buffer_ids);
    const auto reduce_buffer_tv = ir_builder_.create<kir::TensorView>(
        out->dtype(), buffer_domain, MemoryType::Global);

    const std::vector<kir::IterDomain*> new_sync_ids = {
        ir_builder_.create<kir::IterDomain>(zero, sync_size)};
    const auto sync_domain =
        ir_builder_.create<kir::TensorDomain>(new_sync_ids);
    const auto reduce_sync_tv = ir_builder_.create<kir::TensorView>(
        DataType::Int, sync_domain, MemoryType::Global);

    const auto reduce_buffer = ir_builder_.create<kir::Allocate>(
        reduce_buffer_tv, reduce_buffer_tv->memoryType());

    const auto sync_buffer = ir_builder_.create<kir::Allocate>(
        reduce_sync_tv, reduce_sync_tv->memoryType(), nullptr, true);

    const auto grid_reduction_op = (block_reduction_op == nullptr)
        ? ir_builder_.create<kir::ReductionOp>(
              rop->operation(), rop->init(), out, in)
        : block_reduction_op;

    auto grid_reduction = ir_builder_.create<kir::GridReduction>(
        grid_reduction_op, reduce_buffer, sync_buffer);
    grid_reduction->setPredicate(pred);

    pushBack(reduce_buffer);
    pushBack(sync_buffer);
    pushBack(grid_reduction);
  }

  if (!is_block_reduce && !is_grid_reduce) {
    pushBack(ir_builder_.create<kir::BinaryOp>(rop->operation(), out, out, in));
  }
}

void IndexLowering::visit(const kir::BroadcastOp* bop) {
  TORCH_INTERNAL_ASSERT(ir_utils::isTVOp(bop));
  const auto out = lowerDstIndex(bop->out());
  const auto in = lowerSrcIndex(bop->in(), bop->out());
  pushBack(ir_builder_.create<kir::BroadcastOp>(out, in));
}

void IndexLowering::visit(const kir::Allocate* allocate) {
  // TODO(kir): remove the need for const_cast
  pushBack(const_cast<kir::Allocate*>(allocate)); // NOLINT
}

void IndexLowering::visit(const kir::Sync* sync) {
  // TODO(kir): remove the need for const_cast
  pushBack(const_cast<kir::Sync*>(sync)); // NOLINT
}

void IndexLowering::generate(const std::vector<kir::Expr*>& exprs) {
  for (auto expr : exprs) {
    expr->accept(this);
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch

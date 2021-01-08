#include <torch/csrc/jit/codegen/cuda/lower_insert_syncs.h>
#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>

#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

//! Scan through Kernel IR to insert Sync nodes to avoid
//! Write-After-Read (WAR) race condition
//!
class LocalSyncInserter {
  using TvSet = std::unordered_set<const kir::TensorView*>;

 public:
  //! Write-After-Read race conditions are only found within for-loops.
  //! Sync nodes are inserted directly into the for-loops.
  //! The expressions are modified in-place and exprs is const.
  static void insertSyncs(const std::vector<kir::Expr*>& exprs) {
    LocalSyncInserter sync_inserter;
    for (auto expr : exprs) {
      sync_inserter.handle(expr);
    }
  }

  const auto& initial() const {
    return initial_;
  }

  const auto& final() const {
    return final_;
  }

  const auto& all_smem_inputs() const {
    return all_smem_inputs_;
  }

  const auto& all_smem_outputs() const {
    return all_smem_outputs_;
  }

 private:
  // TODO(kir): this is a place where a mutable IR visitor may be appropriate
  void handle(kir::Expr* expr) {
    if (ir_utils::isTVOp(expr)) {
      // For this SyncInserter
      initial_sync_ ? addInputSmemTvs(expr, final_)
                    : addOutputSmemTvs(expr, initial_);

      // For parent SyncInserter
      addOutputSmemTvs(expr, all_smem_outputs_);
      addInputSmemTvs(expr, all_smem_inputs_);
    } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
      handle(ite);
    } else if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      handle(for_loop);
    }
  }

  void handle(kir::IfThenElse* ite) {
    for (auto expr : ite->thenBody().exprs()) {
      handle(expr);
    }
    for (auto expr : ite->elseBody().exprs()) {
      handle(expr);
    }
  }

  void handle(kir::ForLoop* fl) {
    // Track if last op in body is sync in nested for-loop
    bool is_last_op_sync_ = false;
    for (auto expr : fl->body().exprs()) {
      is_last_op_sync_ = false;
      if (expr->isA<kir::Sync>()) {
        initial_sync_ = true;
        final_.clear();
      } else if (expr->isA<kir::ForLoop>()) {
        // Recursively handle nested for-loop
        LocalSyncInserter child_sync_inserter;
        child_sync_inserter.handle(expr);
        const auto& child_inputs = child_sync_inserter.all_smem_inputs();
        const auto& child_outputs = child_sync_inserter.all_smem_outputs();

        // Default - Track all smem inputs / outputs
        all_smem_inputs_.insert(child_inputs.begin(), child_inputs.end());
        all_smem_outputs_.insert(child_outputs.begin(), child_outputs.end());

        if (!initial_sync_) {
          // Parent - None
          if (!child_sync_inserter.initial_sync_) {
            // Child - None
            // Append All Child Outputs to Parent Initial
            initial_.insert(child_outputs.begin(), child_outputs.end());
          } else if (child_sync_inserter.has_war_hazard_sync_) {
            // Child - WAR race
            // Parent first sync
            // Inherit Child Initial / Clear Parent Final
            initial_sync_ = true;
            is_last_op_sync_ = true;
            initial_.insert(
                child_sync_inserter.initial().begin(),
                child_sync_inserter.initial().end());
            final_.clear();
          } else {
            // Child - 1+
            // Parent first sync
            // Inherit Child Initial + Final
            initial_sync_ = true;
            initial_.insert(
                child_sync_inserter.initial().begin(),
                child_sync_inserter.initial().end());
            final_.insert(
                child_sync_inserter.final().begin(),
                child_sync_inserter.final().end());
          }
        } else {
          // Parent - 1+
          if (!child_sync_inserter.initial_sync_) {
            // Child - None
            // Append All Child to Parent Last
            final_.insert(child_inputs.begin(), child_inputs.end());
          } else if (child_sync_inserter.has_war_hazard_sync_) {
            // Child - WAR race
            // Clear Parent Last / Discard Child Initial
            is_last_op_sync_ = true;
            final_.clear();
          } else {
            // Child - 1+
            // Inherit Child Final / Discard Child Initial
            final_.insert(
                child_sync_inserter.final().begin(),
                child_sync_inserter.final().end());
          }
        }
      } else {
        handle(expr);
      }
    }

    // This level of the nested for-loop may not exist in the kernel.
    // However, subsequent levels can exist, so we handle the body of the
    // for-loop first.
    if (!fl->iter_domain()->isThread() && !fl->iter_domain()->isBroadcast()) {
      // Determine if any smem TV is written to at beginning of the for-loop
      // and whether that smem TV is read from at the end of the for-loop
      // Insert new SyncThreads at end of for-loop to prevent WAR race condition
      //
      // TODO: replace __syncthreads with __threadfence for alias ops
      //
      if (detectIntersection(initial_, final_) &&
          !fl->body().exprs().back()->isA<kir::Sync>() && !is_last_op_sync_) {
        has_war_hazard_sync_ = true;
        kir::IrBuilder ir_builder(GpuLower::current()->kernel());
        fl->body().push_back(ir_builder.create<kir::Sync>(true));
      }
    }
  }

  static bool detectIntersection(const TvSet& left, const TvSet& right) {
    for (auto item : left) {
      if (right.find(item) != right.end()) {
        return true;
      }
    }
    return false;
  }

  static void addOutputSmemTvs(const kir::Expr* expr, TvSet& set) {
    for (auto out : expr->outputs()) {
      if (auto tv = dynamic_cast<kir::TensorView*>(out)) {
        if (tv->memoryType() == MemoryType::Shared) {
          set.insert(tv);
        }
      }
    }
  }

  static void addInputSmemTvs(const kir::Expr* expr, TvSet& set) {
    for (auto in : expr->inputs()) {
      if (auto tv = dynamic_cast<kir::TensorView*>(in)) {
        if (tv->memoryType() == MemoryType::Shared) {
          set.insert(tv);
        }
      }
    }
  }

 private:
  // Track Shared Memory Inputs (Reads) for parent for-loop
  TvSet all_smem_inputs_;

  // Track Shared Memory Outputs (Writes) for parent for-loop
  TvSet all_smem_outputs_;

  // Shared Memory Writes at beginning of the for-loop
  // before first SyncThreads
  TvSet initial_;

  // Shared Memory Reads at end of the for-loop
  // Cleared after each SyncThreads
  TvSet final_;

  // Track first sync found in for-loop
  bool initial_sync_ = false;

  // Track sync was inserted for war hazard
  bool has_war_hazard_sync_ = false;
};

class ExprFlattener : private kir::ConstIrVisitor {
 private:
  void handle(kir::Expr* expr) {
    if (expr->isA<kir::ForLoop>() || expr->isA<kir::IfThenElse>()) {
      expr->accept(this);
    } else {
      exprs.push_back(expr);
    }
  }

  void visit(const kir::ForLoop* fl) final {
    for (auto expr : fl->body().exprs()) {
      handle(expr);
    }
  }

  void visit(const kir::IfThenElse* ite) final {
    for (auto expr : ite->thenBody().exprs()) {
      handle(expr);
    }
    for (auto expr : ite->elseBody().exprs()) {
      handle(expr);
    }
  }

 private:
  std::vector<kir::Expr*> exprs;

 public:
  //! Flattens scopes extracting out a single ordered list of exprs.
  static std::vector<kir::Expr*> flatten(std::vector<kir::Expr*> loop_nests) {
    ExprFlattener flattener;
    for (auto expr : loop_nests) {
      flattener.handle(expr);
    }
    return flattener.exprs;
  }
};

class ReadAfterWriteSyncs : public kir::IrVisitor {
 private:
  void handle(kir::Expr* expr) {
    if (!ir_utils::isTVOp(expr) || expr->isA<kir::Allocate>()) {
      expr->accept(this);
      return;
    }

    if (sync_after.front() == expr) {
      sync_after.pop_front();
      // Found that a sync is needed
      TORCH_INTERNAL_ASSERT(expr->outputs()[0]->isA<kir::TensorView>());
      auto out_tv = expr->outputs()[0]->as<kir::TensorView>();

      // Find where a sync needs to be inserted
      // This is very similar to how allocations are placed, simply place sync
      // after the expression instead of placing like allocation where it goes
      // before before.
      // TODO: This may be a common operation, could be worth making a utility
      // out of or saving state for tensor view ID -> for loop
      // TODO: Explicitly test the 3 cases below

      kir::IrBuilder ir_builder(GpuLower::current()->kernel());
      auto sync_expr = ir_builder.create<kir::Sync>();
      int produced_at = ca_maps_.producedAt(out_tv);
      if (produced_at == 0) {
        // Sync should be placed at global scope, after its outer most loop if
        // it has one.
        kir::Expr* place_after = for_loops.size() > 0 ? for_loops[0] : expr;
        // Find location in loop_nests_
        auto place_after_it =
            std::find(loop_nests_.begin(), loop_nests_.end(), place_after);
        TORCH_INTERNAL_ASSERT(
            place_after_it != loop_nests_.end(),
            "Could not figure out where to place synchronization. ",
            "Tried to place after, ",
            toString(place_after, false),
            ", but could not find this expression at the global scope.");
        loop_nests_.insert(place_after_it + 1, sync_expr);
      } else {
        // Find the last loop in computeAt of out_tv, this is the loop where we
        // would place an allocation for out_tv
        auto fuser_tv = out_tv->fuserTv();
        auto lowered_local_id =
            gpu_lower->lowerValue(fuser_tv->axis(produced_at - 1))
                ->as<kir::IterDomain>();

        auto loops_it = std::find_if(
            for_loops.begin(), for_loops.end(), [&](const auto& loop) {
              return this->ca_maps_.areMapped(
                         loop->iter_domain(), lowered_local_id) ||
                  loop->iter_domain()->parallelType() == ParallelType::Unroll;
            });

        TORCH_INTERNAL_ASSERT(loops_it != for_loops.end());

        auto place_in = *loops_it;
        kir::Expr* place_after = nullptr;

        if (loops_it + 1 == for_loops.end()) {
          // Inline allocation, place after expr
          place_after = expr;
        } else {
          // Place allocation after the last computeAt axis
          // TODO: may be more efficient to place after the first non-computeAt
          // axis
          place_after = *(loops_it + 1);
        }

        place_in->body().insert_after(place_after, sync_expr);
      }
    }
  }

  void visit(kir::ForLoop* fl) final {
    for_loops.push_back(fl);
    // Modifying in place, make a copy of the vector
    const std::vector<kir::Expr*> exprs = fl->body().exprs();
    for (auto expr : exprs) {
      handle(expr);
    }
    for_loops.pop_back();
  }

  void visit(kir::IfThenElse*) final {
    TORCH_INTERNAL_ASSERT(
        false,
        "Pass does not support conditional statements, ",
        "this pass should be run before any conditionals are placed in code.");
  }

  // Clear the modify status for all shared memory buffers
  static void cleanSharedMemory(std::unordered_map<kir::Val*, bool>& smem) {
    for (auto& item : smem) {
      item.second = false;
    }
  }

  // Return the status of the shared memory buffer
  // False if TensorView is not shared memory buffer
  bool isModifiedSharedMemory(
      std::unordered_map<kir::Val*, bool>& smem,
      std::vector<kir::Val*> keys) const {
    return std::any_of(keys.begin(), keys.end(), [&smem](kir::Val* key) {
      auto it = smem.find(key);
      if (it != smem.end()) {
        return it->second;
      }
      return false;
    });
  }

  ReadAfterWriteSyncs(std::vector<kir::Expr*> _loop_nests)
      : loop_nests_(_loop_nests),
        gpu_lower(GpuLower::current()),
        ir_builder(gpu_lower->kernel()),
        ca_maps_(GpuLower::current()->caMaps()) {
    // Fusion shared_memory values
    // Tracks if shared memory is modified
    std::unordered_map<kir::Val*, bool> smem;

    // Flatten all the expressions
    auto flattened_exprs = ExprFlattener::flatten(loop_nests_);

    kir::Expr* prev_tv_expr = nullptr;
    for (auto expr : flattened_exprs) {
      if (!ir_utils::isTVOp(expr) || expr->isA<kir::Allocate>()) {
        continue;
      }

      bool need_sync = isModifiedSharedMemory(smem, expr->inputs());
      if (need_sync) {
        TORCH_INTERNAL_ASSERT(
            prev_tv_expr != nullptr,
            "Can't require sync on inputs, however, detected it's needed.");
        sync_after.push_back(prev_tv_expr);
        cleanSharedMemory(smem);
      }

      for (auto out : expr->outputs()) {
        if (out->isA<kir::TensorView>()) {
          if (out->as<kir::TensorView>()->memoryType() == MemoryType::Shared) {
            smem[out] = true;
          }
        }
      }

      prev_tv_expr = expr;
    }

    // Insert read after write syncs
    const std::vector<kir::Expr*> exprs = loop_nests_;
    for (auto expr : exprs) {
      handle(expr);
    }

    TORCH_INTERNAL_ASSERT(
        sync_after.empty(), "Didn't place all required syncs.");
  }

 private:
  std::deque<kir::Expr*> sync_after;

  std::vector<kir::ForLoop*> for_loops;

  std::vector<kir::Expr*> loop_nests_;

  GpuLower* gpu_lower;

  kir::IrBuilder ir_builder;

  const ComputeAtMap& ca_maps_;

 public:
  static std::vector<kir::Expr*> insert(std::vector<kir::Expr*> loop_nests) {
    ReadAfterWriteSyncs inserter(loop_nests);
    return inserter.loop_nests_;
  }
};

} // namespace

std::vector<kir::Expr*> insertRAWThreadSynchronization(
    std::vector<kir::Expr*> exprs) {
  FUSER_PERF_SCOPE("insertRAWThreadSynchronization");
  return ReadAfterWriteSyncs::insert(exprs);
}

std::vector<kir::Expr*> insertWARThreadSynchronization(
    const std::vector<kir::Expr*>& exprs) {
  FUSER_PERF_SCOPE("insertWARThreadSynchronization");
  LocalSyncInserter::insertSyncs(exprs);
  return exprs;
}
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch

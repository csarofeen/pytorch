#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/lower_compute_at_map.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>

#include <memory>
#include <ostream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class TORCH_CUDA_API GpuLower {
  class KernelIrMapper;

 public:
  GpuLower() = default;

  explicit GpuLower(Fusion* fusion) : fusion_(fusion) {
    lower();
  }

  kir::Kernel* kernel() const;

  //! Converts a Fusion IR value into the Kernel IR equivalent
  kir::Val* lowerValue(const Val* val);

  //! Converts a Fusion IR expression into the Kernel IR equivalent
  kir::Expr* lowerExpr(const Expr* expr);

  //! Returns the currently active lowering object
  //! (or nullptr if no lowering is in progress)
  static GpuLower* current();

  const ComputeAtRootDomainMap& caRootMap() const {
    return ca_root_map;
  }

  const ComputeAtMap& caLoopMap() const {
    return ca_loop_map;
  }

  const ComputeAtMap& caIndexMap() const {
    return ca_index_map;
  }

 private:
  void lower();

  // TensorViews are all based on symbolic sizes. When we first initialize them
  // we don't know if they're inputs or outputs which would mean that they have
  // runtime shapes. Intermediate tensors (those not going to global memory) do
  // not have this information. Since we need to have the correct information in
  // the kernel being fetched for shapes, we want to replace input and output
  // tensors to reference the runtime structure containing sizes.
  void replaceSymbolicSizes();

 private:
  // Lowered Kernel IR
  std::unique_ptr<kir::Kernel> kernel_;

  // Fusion IR node to Kernel IR node mapping
  std::unordered_map<const Val*, kir::Val*> kir_val_map_;
  std::unordered_map<const Expr*, kir::Expr*> kir_expr_map_;

  // Some stateful information during lowering

  ComputeAtRootDomainMap ca_root_map;
  ComputeAtMap ca_loop_map;
  ComputeAtMap ca_index_map;

  Fusion* fusion_ = nullptr;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch

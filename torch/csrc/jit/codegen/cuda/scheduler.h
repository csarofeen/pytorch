#pragma once

#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// return true or false on whether given fusion could be scheduled;
TORCH_CUDA_API bool scheduleFusion(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue> inputs);

// TODO: This function is currently a redundant API as I populate a more
// substantial reduction heuristic
// fusion is the input IR that will be modified by this function
TORCH_CUDA_API bool scheduleReduction(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue> inputs);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch

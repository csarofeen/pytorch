
#include <torch/csrc/jit/codegen/cuda/executor_kernel_arg.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <torch/csrc/jit/codegen/cuda/executor.h>

#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

int FusionExecutor::fusion_id_counter_ = 0;

std::string FusionExecutor::getStructuredCode(const std::string& kernel) {
  // generating cuda code;
  std::string code = std::string("namespace ") +
      FusionExecutor::kernelNamespace() + " {\n" +
      executor_utils::kernelPreamble() + kernel + "}\n";

  const char* debug_env = getenv("PYTORCH_CUDA_FUSER_DEBUG");
  if (debug_env && atoi(debug_env)) {
    std::cout << "\n==== codegen output for kernel: " << kernelName()
              << " ====" << std::endl
              << code << std::endl
              << "=====*===============================" << std::endl;
  }

  return code;
}

void FusionExecutor::debugCompileFusionFromStr(
    Fusion* fusion,
    const std::string& code,
    const std::string& name,
    int id,
    CompileOptions options) {
  fusion_ = *fusion;
  FusionGuard fg(&fusion_);
  options_ = options;

  const char* debug_env = getenv("PYTORCH_CUDA_FUSER_DEBUG");
  if (debug_env && atoi(debug_env)) {
    std::cout << "\n==== codegen output for kernel: " << kernelName()
              << " ====" << std::endl
              << code << std::endl
              << "=====*===============================" << std::endl;
  }

  fusion_id_ = id;
  has_random_ = fusion->hasRNG();
  lowered_ = GpuLower(&fusion_);
  compiled_kernel_ = executor_utils::nvrtcCompile(code, name, fusion_id_);
  compiled_ = true;
}

void FusionExecutor::compileFusion(Fusion* fusion, CompileOptions options) {
  TORCH_INTERNAL_ASSERT(
      !fusion->outputs().empty(), "No output found for this kernel, aborting.");

  for (auto out : fusion->outputs()) {
    TORCH_INTERNAL_ASSERT(
        out->getValType() == ValType::TensorView,
        "Output types from fusions that are not tensors are not supported at this point.");
  }

  // Clone the fusion so we can store it
  fusion_ = *fusion;
  FusionGuard fg(&fusion_);
  options_ = options;

  TORCH_INTERNAL_ASSERT(
      options.device.is_cuda(), "Provided device to CUDA fuser is the CPU.");
  max_device_smem =
      at::cuda::getDeviceProperties(options.device.index())->sharedMemPerBlock;

  fusion_id_ = ++fusion_id_counter_;
  has_random_ = fusion->hasRNG();
  has_block_reductions = fusion_.hasBlockReduction();
  has_grid_reductions = fusion_.hasGridReduction();
  has_block_broadcasts = fusion_.hasBlockBroadcast();
  lowered_ = GpuLower(&fusion_);
  const auto kernel = lowered_.getKernel(kernelName());
  const auto structured_code = getStructuredCode(kernel);

  if (lowered_.static_allocations().size() > 0) {
    StatefulExpressionEvaluator static_evaluator(&fusion_);
    unsigned static_smem_size =
        computeSharedMemory(static_evaluator, lowered_.static_allocations());
    TORCH_INTERNAL_ASSERT(
        static_smem_size < max_device_smem,
        "The static shared memory allocation is larger than available memory.");
  }

  compiled_kernel_ = executor_utils::nvrtcCompile(
      structured_code,
      (kernelNamespace() + "::" + kernelName()).c_str(),
      fusion_id_);
  compiled_ = true;
}

namespace {

at::Tensor inferAndAlloc(
    const TensorView* tv,
    StatefulExpressionEvaluator& see,
    const CompileOptions& options,
    bool zero_init = false) {
  std::vector<int64_t> sizes;
  for (auto id : TensorDomain::noReductions(tv->getRootDomain())) {
    auto inferred_val = see.inferValue(id->rawExtent());
    TORCH_INTERNAL_ASSERT(
        inferred_val.has_value(),
        "Could not launch kernel as program could not infer ",
        id->rawExtent(),
        " for the buffer ",
        tv);
    sizes.push_back(inferred_val.value());
  }

  auto at_type = data_type_to_aten(tv->getDataType().value());
  auto tensor_options =
      at::TensorOptions().dtype(at_type).device(options.device);

  if (zero_init) {
    c10::IntArrayRef isizes(sizes);
    return at::zeros(isizes, tensor_options);
  } else {
    c10::IntArrayRef isizes(sizes);
    return at::empty(isizes, tensor_options);
  }
}

} // namespace

uint64_t FusionExecutor::computeSharedMemory(
    StatefulExpressionEvaluator& see,
    const std::vector<kir::Allocate*>& buffers,
    bool align_padding,
    uint64_t total) {
  for (auto smem_alloc : buffers) {
    auto inferred_val = see.inferValue(smem_alloc->size());
    if (inferred_val.has_value()) {
      const uint64_t data_size = dataTypeSize(smem_alloc->buffer_type());
      // Add padding to align dynamic shared memory
      if (align_padding) {
        total = ceilDiv(total, data_size) * data_size;
      }
      total += inferred_val.value() * data_size;
    } else {
      TORCH_INTERNAL_ASSERT(
          false,
          "Failed to evaluate the size ",
          smem_alloc->size(),
          " of shared memory buffer - T",
          smem_alloc->buffer()->name());
    }
  }
  return total;
}

LaunchParams FusionExecutor::computeLaunchParams(
    const at::ArrayRef<IValue>& aten_inputs,
    const LaunchParams& launch_constraints,
    StatefulExpressionEvaluator& see) {
  LaunchParams launch_params;

  // Grab all values that are actually used in the fusion
  auto unordered_vals = DependencyCheck::getAllValsBetween(
      {fusion_.inputs().begin(), fusion_.inputs().end()}, fusion_.outputs());

  // Lets collect all IterDomains that are bound to a thread binding
  std::unordered_map<ParallelType, std::vector<IterDomain*>, TypeHash>
      parallel_iter_domains;

  for (auto val : unordered_vals) {
    if (val->getValType().value() == ValType::TensorView) {
      TensorView* tv = val->as<TensorView>();
      for (auto id : tv->domain()->domain()) {
        if (id->isThread() && !id->isBroadcast()) {
          if (parallel_iter_domains.find(id->getParallelType()) !=
              parallel_iter_domains.end()) {
            parallel_iter_domains.at(id->getParallelType()).push_back(id);
          } else {
            parallel_iter_domains[id->getParallelType()] =
                std::vector<IterDomain*>({id});
          }
        }
      }
    }
  }

  // If any dimension was set in launch constraints we need to run through
  // IterDomains that have been parallelized, and bind those values. Or make
  // sure if they could be inferred the inference matches what was set.
  if (launch_constraints.nBlocks() * launch_constraints.nThreads() != -1) {
    for (auto& entry : parallel_iter_domains) {
      auto p_type = entry.first;
      if (launch_constraints.hasDim(p_type)) {
        auto parallel_ids = entry.second;
        for (auto parallel_id : parallel_ids) {
          auto inferred_val = see.inferValue(parallel_id->rawExtent());
          if (inferred_val.has_value()) {
            // This value could have been infered, make sure it was set right.
            TORCH_CHECK(
                inferred_val.value() == launch_constraints.getDim(p_type) ||
                    launch_constraints.getRawVal(p_type) == -1,
                "inferred that ",
                p_type,
                " should be set to ",
                inferred_val.value(),
                " but launch constraints specified ",
                launch_constraints.getDim(p_type));
          } else {
            // Bind the launch constraint into our evaluation context
            see.safeBind(
                parallel_id->rawExtent(),
                launch_constraints.getDim(entry.first));
            launch_params.bind(launch_constraints.getDim(p_type), p_type);
          }
        }
      }
    }
  }

  // Run through the rest of the parallel IterDomains and infer their size
  for (auto& entry : parallel_iter_domains) {
    auto p_type = entry.first;
    auto parallel_ids = entry.second;
    for (auto parallel_id : parallel_ids) {
      auto val = see.inferValue(parallel_id->rawExtent());
      TORCH_INTERNAL_ASSERT(
          val,
          "Tried to evaluate the extent of ",
          parallel_id,
          " to set launch bounds but could not.");
      launch_params.bind(val.value(), p_type);
    }
  }

  // Calculate Dynamic Shared Memory Size
  // Add workspace for reduction and broadcast
  uint64_t reduction_broadcast_workspace = 0;
  if (has_block_reductions || has_grid_reductions || has_block_broadcasts) {
    // Not using nThreads here since it does not handle uninitialized value
    reduction_broadcast_workspace =
        dataTypeSize(fusion_.getMaximumSmemDataType()) * launch_params.bdimx() *
        launch_params.bdimy() * launch_params.bdimz();
  }

  uint64_t dynamic_smem_size = computeSharedMemory(
      see, lowered_.dynamic_allocations(), true, reduction_broadcast_workspace);

  uint64_t static_smem_size =
      computeSharedMemory(see, lowered_.static_allocations());

  TORCH_INTERNAL_ASSERT(
      (dynamic_smem_size + static_smem_size) < max_device_smem,
      "The total shared memory allocation is larger than available memory.");
  launch_params.setSmem(dynamic_smem_size);

  return launch_params;
}

std::vector<at::Tensor> FusionExecutor::allocGlobalVals(
    StatefulExpressionEvaluator& see) {
  std::vector<at::Tensor> global_buffers;
  for (auto alloc : lowered_.global_allocations()) {
    TORCH_INTERNAL_ASSERT(
        alloc->buffer()->getValType() == ValType::KirTensorView,
        "Cannot allocate global buffers that are not tensors.");
    global_buffers.push_back(inferAndAlloc(
        alloc->buffer()->as<kir::TensorView>()->fuserTv(),
        see,
        options_,
        false));
  }

  for (auto alloc : lowered_.sync_allocations()) {
    TORCH_INTERNAL_ASSERT(
        alloc->buffer()->getValType() == ValType::KirTensorView,
        "Cannot allocate global buffers that are not tensors.");
    global_buffers.push_back(inferAndAlloc(
        alloc->buffer()->as<kir::TensorView>()->fuserTv(),
        see,
        options_,
        true));
  }

  return global_buffers;
}

std::vector<at::Tensor> FusionExecutor::allocOutputs(
    StatefulExpressionEvaluator& see) {
  std::vector<at::Tensor> outputs;
  for (auto output : fusion_.outputs()) {
    TORCH_INTERNAL_ASSERT(
        output->getValType() == ValType::TensorView,
        "Cannot allocate outputs that are not tensors.");
    outputs.push_back(
        inferAndAlloc(output->as<TensorView>(), see, options_, false));
  }
  return outputs;
}

std::vector<at::Tensor> FusionExecutor::runFusion(
    const at::ArrayRef<IValue>& inputs,
    const std::vector<at::Tensor>& outputs,
    const LaunchParams& launch_constraints) {
  TORCH_INTERNAL_ASSERT(
      fusion_id_ > 0, "Cannot run fusion, it was not compiled.");

  FusionGuard fg(&fusion_);

  executor_utils::validateKernelInputs(&fusion_, inputs, options_.device);

  c10::DeviceGuard dg(options_.device);
  auto stream = at::cuda::getCurrentCUDAStream();

  StatefulExpressionEvaluator evaluator =
      executor_utils::statefulBindInputs(inputs, &fusion_);

  LaunchParams launch_params =
      computeLaunchParams(inputs, launch_constraints, evaluator);

  std::vector<at::Tensor> alloced_outputs = outputs;
  if (outputs.empty() || outputs.size() != fusion_.outputs().size()) {
    alloced_outputs = allocOutputs(evaluator);
  }

  executor_utils::validateKernelOutputs(
      &fusion_, alloced_outputs, options_.device);

  KernelArgumentHolder kernel_arguments;
  kernel_arguments.push(inputs);
  kernel_arguments.push(alloced_outputs);
  auto buffers = allocGlobalVals(evaluator);
  kernel_arguments.push(buffers);

  if (has_random_) {
    const auto rand_offset = 4 *
        (std::ceil(
             alloced_outputs[0].numel() / (4.0 * 128 * launch_params.gdimx())) +
         1);
    kernel_arguments.appendPhiloxRNGSeed(rand_offset);
  }

  AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuLaunchKernel(
      compiled_kernel_.function,
      launch_params.gdimx(),
      launch_params.gdimy(),
      launch_params.gdimz(),
      launch_params.bdimx(),
      launch_params.bdimy(),
      launch_params.bdimz(),
      launch_params.smem(),
      stream,
      kernel_arguments.getBuffer(),
      nullptr));
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));

  return alloced_outputs;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch

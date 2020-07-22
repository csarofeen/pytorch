#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/parser.h>
#include <torch/csrc/jit/codegen/cuda/scheduler.h>
#include <torch/csrc/jit/codegen/cuda/shape_inference.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/interpreter.h>

#include <unordered_map>

#include <c10/core/DeviceType.h>

#include <torch/csrc/jit/codegen/cuda/manager.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {
c10::Device getDevice(const at::ArrayRef<IValue>& inputs) {
  // find device in inputs.
  for (const auto& input : inputs) {
    if (input.isTensor()) {
      auto dev = input.toTensor().device();
      TORCH_INTERNAL_ASSERT(
          dev.is_cuda(), "Could only fuser operations on cuda device");
      return dev;
    }
  }
  TORCH_INTERNAL_ASSERT(
      false, "Could not detect device of inputs to a fusion.");
}

// CudaFusionManager holds a FusionExecutor and handles all interfacing
// including compilation and execution.
//
// We cache two maps here:
//   a. string of graph -> kernel_id
//   b. kernel_id -> FusionExecutor
//
// This allows FusionExecutor reuse across nodes;
class CudaFusionManager {
 public:
  static CudaFusionManager& getManager() {
    static CudaFusionManager cuda_fusion_manager_;
    return cuda_fusion_manager_;
  };

  // TODO: I'm assuming we have stride information in `graph->toString`
  //       We need to make sure stride information is in the final string, as we
  //       want to AVOID kernel reuse between different fusion_node, unless they
  //       have identical contiguity information! (So identical stride + shape
  //       is even more restricting in a good way)
  int32_t registerOrGetCacheId(std::shared_ptr<Graph>& graph) {
    std::lock_guard<std::mutex> guard(mutex_);

    // prepare graph for lowering;
    // We should not call `EraseShapeInformation(graph);`, graph representation
    // does not incorporate static sizes, but just rank of input tensors, which
    // is exactly what we wanted.
    std::cout << "\nprior to canonical\n" << *graph << std::endl;
    Canonicalize(graph, false);
    std::cout << "\nafter canonical\n" << *graph << std::endl;
    auto repr = graph->toString(false);

    // create new graph_cache_ entry;
    if (graph_cache_.count(repr) == 0) {
      int32_t kernel_id = getNextUniqueID();
      graph_cache_[repr] = kernel_id;
    }
    return graph_cache_[repr];
  };

  std::vector<at::Tensor> runFusionNode(
      int32_t kernel_id,
      std::shared_ptr<Graph>& graph,
      const at::ArrayRef<IValue> inputs) {
    std::lock_guard<std::mutex> guard(mutex_);

    auto inputs_vec = dimCollapseInputs(graph, inputs);
    const at::ArrayRef<IValue> inputs_ref = inputs_vec;

    FusionExecutor* fe;
    if (kernel_cache_.find(kernel_id) == kernel_cache_.end()) {
      // search kernel cache failed, we need to codegen new kernel for given
      // inputs;

      auto copy = dimCollapseGraph(graph);
      auto fusion = parseJitIR(copy);

      // TODO: update the API to let `scheduleFusion` consume & return a fusion
      // magic scheduler updates fusion instance via transformation and setup
      // launch configurations;
      scheduleFusion(fusion.get(), inputs_ref);

      CompileOptions options;
      options.device = getDevice(inputs_ref);

      kernel_cache_[kernel_id] = std::make_unique<FusionExecutor>();
      kernel_cache_[kernel_id]->compileFusion(fusion.get(), options);
    }

    fe = kernel_cache_[kernel_id].get();
    return dimCollapseOutputs(graph, fe->runFusion(inputs_ref));
  }
 
 private:
  // TODO: Dimension collapsing should be abstracted out and integrated into
  // graph caching.

  // Dimension collapsing only applicable to profiling executor at this moment
  bool graphHasReduction(const std::shared_ptr<Graph>& graph) {
    for (const auto& n : graph->nodes()) {
      if (isReductionNode(n)) {
        return true;
      }
    }
    return false;
  }

  TensorTypePtr extractDimensionCollapse(const std::shared_ptr<Graph>& graph) {
    // run over inputs to extract common types;
    TensorTypePtr acc_type = TensorType::get();
    for (const auto& input : graph->inputs()) {
      // only check tensor types;
      if (auto input_type = input->type()->cast<TensorType>()) {
        //printf("\nlook at inputs:\n");
        //debugPrint(input_type);
        if (!input_type->dim().has_value()) {
          printf("early termination");
          // early termination when detecting undefined tensor;
          return TensorType::get()->withUndefined();
        }
        if (acc_type->dim().has_value()) {
          // TODO: I think merge cannot handle broadcast - Go verify it later;
          acc_type = acc_type->merge(input_type);
        } else {
          acc_type = input_type;
        }
        //printf("\nacc type\n");
        //debugPrint(acc_type);
      }
    }
    return acc_type;
  }

  void debugPrint(TensorTypePtr type) {
    if (auto sizes = type->symbolic_sizes().sizes()) {
      printf("size: ");
      //for (const auto& shape_symbol : sizes.value()) {
      int rank = static_cast<int>(sizes->size());
      for (int i = 0; i < rank; i++) {
        const auto& shape_symbol = sizes.value()[i];
        if (shape_symbol.is_static()) {
          printf("%ld, ", shape_symbol.static_size());
        } else {
          printf("s(%ld), ", *reinterpret_cast<const int64_t*>(&shape_symbol));
        }
      }
    } else {
      printf("no size available\n");
    }
    if (const auto& stride_properties = type->stride_properties().sizes()) {
      int rank = static_cast<int>(stride_properties->size());
      printf("\nstride: ");
      for (int i = 0; i < rank; i++) {
        if (auto val = (*stride_properties)[i]->stride_) {
          printf("%ld, ", val.value());
        } else {
          printf("?, ");
        }
      }
      printf("\nstride index: ");
      for (int i = 0; i < rank; i++) {
        if (auto val = (*stride_properties)[i]->stride_index_) {
          printf("%ld, ", val.value());
        } else {
          printf("?, ");
        }
      }
      printf("\ncontiguous: ");
      for (int i = 0; i < rank; i++) {
        if (auto val = (*stride_properties)[i]->contiguous_) {
          printf("%d, ", val.value());
        } else {
          printf("?, ");
        }
      }
    } else {
      printf("no stride properties available\n");
    }
  }

  std::vector<std::vector<int>> getCollapsingScheme(TensorTypePtr type) {
    // `collapsed_dims` is the returned dimension collapsing strategy;
    std::vector<std::vector<int>> collapsed_dims;

    auto sizes = type->symbolic_sizes().sizes();
    auto stride_properties = type->stride_properties().sizes();

    TORCH_INTERNAL_ASSERT(sizes.has_value() && stride_properties.has_value(),
        "unknown sizes or stride_properties, collapsing shouldn't happen");

    // TODO: reuse this;
    const int rank = static_cast<int>(sizes->size());

    // stores axes with stride_index;
    std::set<int> ordered_axes;

    // TODO: this does not support broadcast yet;
    for (int i = 0; i < rank; i++) {
      if (auto index = (*stride_properties)[i]->stride_index_) {
        ordered_axes.insert(*index);
			}
    }

    collapsed_dims.emplace_back();
    int num_collapsed_dims = 0;
    int unallocated_axis = 0;
    for (int i = rank-1; i >= 0; i--) {
      if (auto index = (*stride_properties)[i]->stride_index_) {
        // pushing axis index to current entry in collapsed_dims;
        collapsed_dims.back().emplace_back(*index);
        // we can not collapse fasted changing dimension;
        if (i != 0) {
          // TODO: exclude reduction axes from collapsing when the support is
			  	//       added;
 			    if ((*stride_properties)[i]->contiguous_.has_value() &&
              (*stride_properties)[i]->contiguous_.value()) {
            printf("\ncollaping %d", i);
            // contiguous flag is true for non-fast-changing-dimension (non-fcd)
            // we could collapse it with neighboring dimension, hence we
 			  		// increase the counter.
			  		num_collapsed_dims++;
          } else {
            printf("\nnot collaping %d", i);
            // non-contiguous dimension expected next, push a new entry to collapsed_dims;
            collapsed_dims.emplace_back();
          }
 			  }
      } else {
        // no designated axis for this slot, so we push an axis with no order;
        while (ordered_axes.count(unallocated_axis) != 0) {
          ++unallocated_axis;
        }
        collapsed_dims.back().emplace_back(unallocated_axis++);
        collapsed_dims.emplace_back();
      }
    }
    return collapsed_dims;
  }
    
  at::Tensor dimCollapseInput(
			const at::Tensor& tensor,
  		const std::vector<std::vector<int>>& dim_col_strategy) {
    int collapsed_rank = dim_col_strategy.size();
    int rank = tensor.dim();

    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;

    for (const auto& dims : dim_col_strategy) {
      int size = 1;
      for (int index : dims) {
        size *= tensor.size(index); // accumulate size
      }
      sizes.emplace_back(size);
      strides.emplace_back(tensor.stride(dims.back())); // set the stride
    }
 		// return tensor with collapsed dimensions
    return tensor.as_strided(sizes, strides);
  }

  std::vector<IValue> dimCollapseInputs(
      std::shared_ptr<Graph>& graph,
      const at::ArrayRef<IValue> inputs) {
    if (!IsNewExecutorEnabled() || graphHasReduction(graph)) {
      return inputs.vec();
    }
    auto acc_type = extractDimensionCollapse(graph);

    printf("\nacc_type");
    debugPrint(acc_type);

    if (!acc_type->dim().has_value()) {
      return inputs.vec();
    }

    auto strategy = getCollapsingScheme(acc_type);
    printf("\n ==== strategy");
    for (const auto& collapsed_dims : strategy) {
      printf("\n\tdim: ");
      for (const auto& dim : collapsed_dims) {
        printf("%d, ", dim);
      }
    }
    std::vector<IValue> collapsed_inputs;
    for (const auto& input : inputs) {
      if (input.isTensor()) {
        collapsed_inputs.emplace_back(dimCollapseInput(input.toTensor(), strategy));
      } else {
        collapsed_inputs.emplace_back(input);
      }
    }
    return collapsed_inputs;
    //return inputs.vec();
  }

  // TODO: we are currently using output types in `graph` in order to restore
  //       sizes from a collapsed dimension.
  //       This is not sufficient though, given that symbolic shape could only
  //       be resolved at run-time. We need to use shape inference (in the
  //       context) in order to get the complete output tensor shapes prior to
  //       dimension collapsing.
  std::vector<at::Tensor> dimCollapseOutputs(
      const std::shared_ptr<Graph>& graph,
      const std::vector<at::Tensor> outputs) {
    if (!IsNewExecutorEnabled() || graphHasReduction(graph)) {
      return outputs;
    }
    auto acc_type = extractDimensionCollapse(graph);
    auto strategy = getCollapsingScheme(acc_type);

    if (!acc_type->dim().has_value()) {
      return outputs;
    }

    std::vector<at::Tensor> uncollapsed_outputs;
    TORCH_INTERNAL_ASSERT(outputs.size() == graph->outputs().size());
    for (int i = 0; i < static_cast<int>(outputs.size()); i++) {
      auto output_type = graph->outputs()[i]->type()->cast<TensorType>();
      TORCH_INTERNAL_ASSERT(output_type->isComplete());

      size_t rank = *output_type->dim();

      std::vector<int64_t> sizes(rank);
      std::vector<int64_t> strides(rank);
      int64_t cur_stride = static_cast<int64_t>(*output_type->numel());

      // we go from slowest to fastest;
      for (const auto& dims : strategy) {
        for (int index : dims) {
          int64_t cur_size = *output_type->sizes()[index];
          sizes[index] = cur_size;
          cur_stride /= cur_size;
          strides[index] = cur_stride;
        }
      }
      uncollapsed_outputs.emplace_back(outputs[i].as_strided(sizes, strides));
    }
    return uncollapsed_outputs;
  }

  std::shared_ptr<Graph> dimCollapseGraph(std::shared_ptr<Graph>& graph) {
    if (!IsNewExecutorEnabled() || graphHasReduction(graph)) {
      return graph->copy();
    }
    auto acc_type = extractDimensionCollapse(graph);
    auto strategy = getCollapsingScheme(acc_type);

    if (!acc_type->dim().has_value()) {
      return graph->copy();
    }

    std::shared_ptr<Graph> copy = graph->copy();
    // TODO: copy over size 1 when we add support for broadcasting;
    // we only need to modify rank
    auto type_transform_fn = [&](TensorTypePtr type) {
      return type->withDim(strategy.size());
    };

    for (auto input : copy->inputs()) {
      if (auto input_type = input->type()->cast<TensorType>()) {
        input->setType(type_transform_fn(input_type));
      }
    }
    return copy;
  }

 private:
  std::mutex mutex_;

  void runCudaKernel(
      int32_t key,
      const std::vector<int>& contiguity_tag,
      const c10::Device){};

  int32_t getNextUniqueID() {
    return next_unique_id_++;
  };

  std::unordered_map<std::string, int32_t> graph_cache_;
  std::unordered_map<int64_t, std::unique_ptr<FusionExecutor>> kernel_cache_;

  int32_t next_unique_id_ = 0;
};

} // namespace

void compileCudaFusionGroup(Node* fusion_node) {
  TORCH_CHECK(
      fusion_node->kind() == prim::CudaFusionGroup,
      "Only prim::CudaFusionGroup can be compiled");
  if (fusion_node->hasAttribute(attr::cache_id)) {
    TORCH_WARN("Double registration of CudaFusionGroup on CudaFusionManager");
  }
  int32_t fusion_cache_id =
      CudaFusionManager::getManager().registerOrGetCacheId(
          fusion_node->g(attr::Subgraph));
  fusion_node->i_(attr::cache_id, fusion_cache_id);
}

void runCudaFusionGroup(const Node* fusion_node, Stack& stack) {
  TORCH_CHECK(
      fusion_node->kind() == prim::CudaFusionGroup,
      "prim::CudaFusionGroup expected");
  // TODO: should we support runtime compilation with updated dynamic shape;
  //       shape inference would be needed so we can allocate output;
  TORCH_CHECK(
      fusion_node->hasAttribute(attr::cache_id),
      "node prim::CudaFusionGroup has not been compiled yet");
  int32_t kernel_id = fusion_node->i(attr::cache_id);

  // Currently we just construct I/O tensors for static graph;
  std::shared_ptr<Graph> graph = fusion_node->g(attr::Subgraph)->copy();

  auto execute_lambda = [&]() {
    const auto nInputs = graph->inputs().size();
    at::ArrayRef<IValue> inputs = last(stack, nInputs);

    // Only needed if we are doing codegen
    // if no shape information available, we feed current shape into the kernel;
    // This is needed because our current broadcast on size-1 dimension 
    if (!IsNewExecutorEnabled()) {
      EraseShapeInformation(graph);
      std::cout << "\nerased shape\n" << *graph << std::endl;
      for (size_t i = 0; i < nInputs; i++) {
        graph->inputs()[i]->setType(inputs[i].type());
      }
      // Type propagation that's here just to cover corner case, incase type
      // propagation failed in the original subgraph. We currently need output
      // types in order to support fp16, where we cast input to fp32 and output
      // back to fp16.
      ShapeTypePropagate(graph);
      std::cout << "\npropogated shape\n" << *graph << std::endl;
    }

    auto outputs =
        CudaFusionManager::getManager().runFusionNode(kernel_id, graph, inputs);

    drop(stack, inputs.size());
    stack.insert(
        stack.end(),
        std::make_move_iterator(outputs.begin()),
        std::make_move_iterator(outputs.end()));
  };

  const char* disable_fb_env = getenv("PYTORCH_CUDA_FUSER_DISABLE_FALLBACK");
  int disable_fb_flag = disable_fb_env ? atoi(disable_fb_env) : 0;
  if (disable_fb_flag) {
    execute_lambda();
  } else {
    try {
      execute_lambda();
    } catch (...) {
      TORCH_WARN(
          "FALLBACK path is taken. This is an indication that codegen"
          "Failed for some reason. To debug try disable codegen fallback path"
          "via setting the env variable"
          "`export PYTORCH_CUDA_FUSER_DISABLE_FALLBACK=1`");
      EraseShapeInformation(graph);
      InterpreterState{Code(graph, "fallback_cuda_fuser")}.run(stack);
    }
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch

#include <torch/csrc/jit/codegen/cuda/scheduler.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/parser.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>

#include <ATen/cuda/CUDAContext.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

constexpr int kUnrollFactor = 4;

namespace {

std::vector<int> reductionAxes(TensorView* tv) {
  size_t n_dims = tv->nDims();
  std::vector<int> reduction_axes;
  for (size_t i = 0; i < n_dims; i++) {
    if (tv->axis(i)->isReduction()) {
      reduction_axes.emplace_back(i);
    }
  }
  return reduction_axes;
}

// coalesces all reduction to the right side and returns total number of
// reduction axes
size_t coalescReduction(TensorView* tv) {
  auto reduction_axes = reductionAxes(tv);
  size_t n_dims = tv->nDims();
  std::unordered_map<int, int> coalesc_permute;
  for (size_t i = 0; i < reduction_axes.size(); i++) {
    size_t new_pos = i + n_dims - reduction_axes.size();
    if (new_pos == reduction_axes[i]) {
      break;
    } else {
      coalesc_permute[reduction_axes[i]] = new_pos;
    }
  }
  if (!coalesc_permute.empty()) {
    tv->reorder(coalesc_permute);
  }
  return reduction_axes.size();
}

} // namespace

// This one is a total mess and it should go.
bool scheduleFusion(Fusion* fusion, const at::ArrayRef<c10::IValue> inputs) {
  FusionGuard fg(fusion);
  // maybe has_reduction for scheudling should be done on a per output tensor
  // basis.
  const bool has_reduction = fusion->hasReduction();
  const bool disable_unroll = fusion->hasRNG();
  bool fcd_reduction = false;

  for (auto out_val : fusion->outputs()) {
    auto out = out_val->as<TensorView>();
    if (has_reduction) {
      // TODO: this scheduling only works for a single reduction operation in
      //       the fusion, in this case we can coalesc all reduction axes and
      //       merge them together. (same applies to iteration axes)
      // TODO: does this work for multiple outputs?

      // query if fastest changing dimension (FCD) is a reduction
      fcd_reduction = out->axis((int)out->nDims() - 1)->isReduction();

      // We coalesc all reduction axes to the right;
      size_t num_reduction_axes = coalescReduction(out);

      // Merge all iteration dimensions
      while (out->nDims() > num_reduction_axes + 1) {
        out->merge(0, 1);
      }
      // Merge all reduction dimensions
      while (out->nDims() > 2) {
        out->merge(1, 2);
      }
    } else {
      // Merge all dimensions because we're only supporting pointwise
      while (out->nDims() > 1)
        out->merge(0, 1);
    }
  }

  if (has_reduction) {
    // Run through outputs, grab all inputs of outputs
    // squeeze with computeAt to set overall structure.
    for (auto output : fusion->outputs()) {
      if (output->getValType() != ValType::TensorView)
        continue;
      TensorView* out_tv = output->as<TensorView>();

      // launch configuratoin.
      TensorView* intermediate = nullptr;
      if (fcd_reduction) {
        out_tv->split(-1, kFcdReductionThreadX);
        // necessary to avoid dynamic allocation on intermediates;
        intermediate = out_tv->rFactor({-2});
      } else {
        // TODO: we don't need a full warp here, this should be determined by
        //       element data type
        out_tv->split(0, kNonFcdReductionThreadX);
        out_tv->split(
            -1, kNonFcdReductionThreadY); // necessary to avoid dynamic
                                          // allocation on intermediates;
        intermediate = out_tv->rFactor({-2});
      }
      for (Val* inp : fusion->inputsOf(output)) {
        // scheduling of inputs shouldn't change with different fcd_reduction
        if (inp->getValType().value() == ValType::TensorView) {
          inp->as<TensorView>()->computeAt(intermediate, -1);
        }
      }
      // scheduling of inputs shouldn't change with different fcd_reduction
      intermediate->computeAt(out_tv, -2);
      if (fcd_reduction) {
        out_tv->axis(0)->parallelize(ParallelType::BIDx);
      } else {
        out_tv->axis(0)->parallelize(ParallelType::BIDx);
        out_tv->axis(1)->parallelize(ParallelType::TIDx);
      }
    }
    // Run through all values, unroll, and bind their axes
    for (auto val : fusion->vals()) {
      if (val->getValType().value() != ValType::TensorView ||
          fusion->hasInput(val))
        continue;
      TensorView* tv = val->as<TensorView>();
      if (fcd_reduction) {
        tv->axis(-1)->parallelize(ParallelType::TIDx);
      } else {
        tv->axis(-1)->parallelize(ParallelType::TIDy);
      }
    }

    TensorView* out0 = fusion->outputs()[0]->as<TensorView>();
    int ndim = (int)out0->nDims();
    Val* numel = new Int(1);
    for (int i = 0; i < ndim; i++) {
      if (out0->axis(i)->isBlockDim()) {
        numel = mul(numel, out0->axis(i)->rawExtent());
      }
    }
  } else {
    // Run through outputs, grab all inputs of outputs
    // squeeze with computeAt to set overall structure.
    for (auto output : fusion->outputs()) {
      if (output->getValType() != ValType::TensorView)
        continue;
      TensorView* out_tv = output->as<TensorView>();

      // Split into 128 which will be bockDim.x
      out_tv->split(0, kPwThreadX);
      // Split by another 4 which will be our unroll factor
      auto ur_factor = disable_unroll ? 1 : kUnrollFactor;
      if (!disable_unroll) {
        out_tv->split(0, ur_factor);
      }
    }

    for (auto output : fusion->outputs()) {
      if (output->getValType() != ValType::TensorView)
        continue;
      TensorView* out_tv = output->as<TensorView>();
      for (Val* inp : fusion->inputsOf(output)) {
        if (inp->getValType().value() == ValType::TensorView)
          inp->as<TensorView>()->computeAt(out_tv, 1);
      }
      out_tv->axis(0)->parallelize(ParallelType::BIDx);
    }

    // Run through all values, unroll, and bind their axes
    for (auto val : fusion->vals()) {
      if (val->getValType().value() != ValType::TensorView ||
          fusion->hasInput(val))
        continue;
      TensorView* tv = val->as<TensorView>();

      // Should be true for all intermediates, but if one isn't hooked
      // up right, skip it and hope for the best for now
      if (!disable_unroll && tv->nDims() == 3) {
        tv->axis(-2)->parallelize(ParallelType::Unroll);
        tv->axis(-1)->parallelize(ParallelType::TIDx);
      } else {
        if (tv->nDims() == 2)
          tv->axis(-1)->parallelize(ParallelType::TIDx);
      }
    }
    TensorView* out0 = fusion->outputs()[0]->as<TensorView>();
    int ndim = (int)out0->nDims();
    Val* numel = new Int(1);
    for (int i = 0; i < ndim; i++) {
      if (out0->axis(i)->isBlockDim()) {
        numel = mul(numel, out0->axis(i)->rawExtent());
      }
    }
  }
  return true;
}

namespace {
inline int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

// Largest Power of 2 less-than n
inline int last_pow2(int n) {
  n |= (n >>  1);
  n |= (n >>  2);
  n |= (n >>  4);
  n |= (n >>  8);
  n |= (n >> 16);
  return std::max(1, n - (n >> 1));
}

// Parameters the Reduction Heuristic Generates to describe
// the optimial schedule
struct ReductionParams {
  // Reduction Blocking
  int grid_dim_x_        = 1;
  int grid_dim_y_        = 1;
  int block_dim_x_       = 1;
  int block_dim_y_       = 1;

  // Reduction Attributes
  bool fastest_dim_      = true;
  bool cross_warp_       = false;
  bool cross_block_      = false;
  bool mul_reds_per_blk_ = false;
};

ReductionParams reductionHeuristic(int outer_dim, int inner_dim, bool red_on_fastest_dim) {
  ReductionParams rparams;
  rparams.fastest_dim_ = red_on_fastest_dim;

  // 1. Initial Assumptions

  // Evaluate Dimensions of Reduction TensorView
  TORCH_INTERNAL_ASSERT(outer_dim > 0 && inner_dim > 0);
  int red_inputs  = outer_dim * inner_dim;
  int red_outputs = (rparams.fastest_dim_ ? outer_dim : inner_dim);
  int red_elems   = (rparams.fastest_dim_ ? inner_dim : outer_dim);

  // 2. Initial Definition of Block Dimensions

  // Is fastest dimension a reduction dimension?
  if (rparams.fastest_dim_) {
    rparams.block_dim_x_ = red_elems;
    rparams.block_dim_y_ = red_outputs;
  } else {
    rparams.block_dim_x_ = red_outputs;
    rparams.block_dim_y_ = red_elems;
  }

  // 3. Applying Power of 2 Blocking based on the Maximum Number of threads

  constexpr int MAX_NUM_THREADS = 512;
  constexpr int VECTOR_SIZE     = 4;
  int num_threads = (rparams.fastest_dim_ ? MAX_NUM_THREADS : MAX_NUM_THREADS / VECTOR_SIZE);
  int DEVICE_WARP_SIZE = at::cuda::warp_size();

  if (rparams.block_dim_x_ < num_threads)
  rparams.block_dim_x_ = last_pow2(rparams.block_dim_x_);
  else
  rparams.block_dim_x_ = num_threads;

  if (rparams.block_dim_y_ < num_threads)
  rparams.block_dim_y_ = last_pow2(rparams.block_dim_y_);
  else
  rparams.block_dim_y_ = num_threads;

  int block_dim_x_prev = rparams.block_dim_x_;
  rparams.block_dim_x_ = std::min(rparams.block_dim_x_, DEVICE_WARP_SIZE);
  rparams.block_dim_y_ = std::min(rparams.block_dim_y_,
                                  num_threads / rparams.block_dim_x_);
  rparams.block_dim_x_ = std::min(block_dim_x_prev,
                                  num_threads / rparams.block_dim_y_);

  // 4. Distributing work across a block

  // Magic numbers of calculations allowed per thread.
  constexpr int MIN_VALUES_PER_THREAD = 16;
  constexpr int MAX_VALUES_PER_THREAD = 256;

  int inputs_consumed_per_block_iter = 1;
  int red_elems_per_thread = red_elems;

  int outputs_produced_per_block_iter = 1;
  bool reduce_inputs_across_warps = false;

  // Reduction is performed across warp threads (cross-thread reduction)
  if (rparams.fastest_dim_) {
    inputs_consumed_per_block_iter *= rparams.block_dim_x_;
    red_elems_per_thread = ceil_div(red_elems_per_thread, inputs_consumed_per_block_iter);
  // Warp threads are applied across the output
  } else {
    outputs_produced_per_block_iter *= rparams.block_dim_x_;
  }

  // Decision to do a cross-warp reduction per block
  if (    red_elems_per_thread >= (rparams.block_dim_y_ * MIN_VALUES_PER_THREAD)
       || red_elems_per_thread >= MAX_VALUES_PER_THREAD                          )
  {
    inputs_consumed_per_block_iter *= rparams.block_dim_y_;
    red_elems_per_thread = ceil_div(red_elems_per_thread, rparams.block_dim_y_);
    reduce_inputs_across_warps = true;
    rparams.cross_warp_ = true;
  // Do multiple reductions per block
  } else {
    rparams.mul_reds_per_blk_ = true;
    outputs_produced_per_block_iter *= rparams.block_dim_y_;
  }

  // 5. Distributing work across blocks

  int DEVICE_MAX_THREADS_PER_MULTIPROCESSOR = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor;
  int DEVICE_MULTIPROCESSOR_COUNT = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  int blocks_per_sm = DEVICE_MAX_THREADS_PER_MULTIPROCESSOR / (rparams.block_dim_x_ * rparams.block_dim_y_);
  int target_grid_size = DEVICE_MULTIPROCESSOR_COUNT * blocks_per_sm;

  //Setting the number of blocks based on the number of outputs
  //grid_dim_x = ceil_div(red_outputs / (red_on_fastest_dim ? 1 : 4), outputs_produced_per_block_iter);
  rparams.grid_dim_x_ = ceil_div(red_outputs , outputs_produced_per_block_iter);

  // Cross-block reductions (if necessary)
  if (    reduce_inputs_across_warps
       && red_elems_per_thread >= MAX_VALUES_PER_THREAD
       && rparams.grid_dim_x_ <= target_grid_size       )
  {
    int blks_per_out_1  = ceil_div(target_grid_size, rparams.grid_dim_x_);
    int blks_per_out_2  = ceil_div(red_elems_per_thread, MIN_VALUES_PER_THREAD);
    int blks_per_out_3  = ceil_div(red_elems_per_thread, MAX_VALUES_PER_THREAD);
    int blks_per_output = std::max(std::min(blks_per_out_1, blks_per_out_2), blks_per_out_3);

    rparams.grid_dim_y_ = std::max(1, blks_per_output);
    //If a cross-block reduction was generated
    if (blks_per_output > 1) {
      rparams.cross_block_ = true;
    //  inputs_consumed_per_block_iter *= blks_per_output;
    //  red_elems_per_thread = ceil_div(red_elems_per_thread, inputs_consumed_per_block_iter);
    }
  }

  const char* debug_env = getenv("PYTORCH_CUDA_FUSER_RED_SCHED_DEBUG");
  if (debug_env && atoi(debug_env)) {
    std::cout << "\n===== Reduction Parameters ========" << std::endl
              << "Inputs:" << std::endl
              << "\tRed Elems: " << red_elems << " Red Outputs: " << red_outputs
              << " Red On Fastest Dim? " << red_on_fastest_dim << std::endl
              << "Reduction Characteristics:" << std::endl
              << "\tMultiple Reds Per Block? " << rparams.mul_reds_per_blk_
              << " Cross Warp? " << rparams.cross_warp_ << " Cross Block? "
              << rparams.cross_block_ << std::endl
              << "Recommended Blocking:" << std::endl
              << "\tGridX: " << rparams.grid_dim_x_
              << " GridY: " << rparams.grid_dim_y_
              << " BlckX: " << rparams.block_dim_x_
              << " BlckY: " << rparams.block_dim_y_ << std::endl
              << "====================================" << std::endl;
  }

  return rparams;
}
} // anonymous namespace

bool scheduleReduction(Fusion* fusion, const at::ArrayRef<c10::IValue> inputs) {
  FusionGuard fg(fusion);

  if (!fusion->hasReduction()) {
    return c10::nullopt;
  }

  // Find Reduction TensorView
  TensorView* red_tv = nullptr;
  for (auto &expr : fusion->exprs(/*from_outputs_only*/true)) {
    if(expr->type() == ExprType::ReductionOp) {
      red_tv = static_cast<TensorView*>(expr->output(0));
    }
  }
  if (red_tv == nullptr) // No reduction found
    return false;

  EvaluationContext eval_context(fusion);

  // I am making some basic assumptions
  // 1.) I am only binding Tensor Dimension sizes.  I am not binding scalar values.
  // 2.) I am only binding the IterDomain.extent().  Do I need to worry about the start?
  for(size_t i = 0; i < inputs.size(); ++i) {
    if(inputs[i].type()->kind() == c10::TypeKind::TensorType) {
      TensorView* tv = static_cast<TensorView*>(fusion->inputs()[i]);
      size_t dims = tv->getRootDomain().size();
      for(size_t j = 0; j < dims; ++j) {
        IterDomain* id = tv->getRootDomain()[j];
        eval_context.bind(id->extent(), inputs[i].toTensor().size(j));
      }
    }
  }

  // Evaluate Dimensions of Reduction TensorView
  auto red_ids = red_tv->domain()->domain();
  std::vector<Int::ScalarType> red_dims(red_ids.size(), 0);
  int red_idx = 0;
  int red_outputs = 1;
  int red_elems   = 1;

  for(size_t i = 0; i < red_ids.size(); ++i) {
    red_dims[i] = ExpressionEvaluator::evaluate(red_ids[i]->extent(), &eval_context).value();
    if (red_ids[i]->isReduction()) {
      red_idx = i;
      red_elems *= red_dims[i];
    } else {
      red_outputs *= red_dims[i];
    }
  }
  bool red_on_fastest_dim = red_idx == (red_dims.size()-1);

  ReductionParams rparams = reductionHeuristic(
                              (red_on_fastest_dim ? red_outputs : red_elems),
                              (red_on_fastest_dim ? red_elems : red_outputs),
                              red_on_fastest_dim
                            );

  // Heuristic Definition
  // TODO: Need to factor in unrolling
  // TODO: Need to get rid of magic numbers.  These should be defined elsewhere.
  if (rparams.fastest_dim_) {
    // Initially I am not going to bother with cross-block reductions or
    // letting a block do multiple reductions to make this simple!

    // Do multiple reductions per block
  if (rparams.mul_reds_per_blk_) {
      red_tv->split(-1, rparams.block_dim_x_);
      // Split Grid dimension to get multiple reds per block
      red_tv->split(0, rparams.block_dim_y_);

      auto red_tv_rf = red_tv->rFactor({-2, -3});
      red_tv_rf->computeAt(red_tv, 1);

      red_tv->axis(0)->parallelize(ParallelType::BIDx);
      red_tv->axis(1)->parallelize(ParallelType::TIDy);
      red_tv->axis(-1)->parallelize(ParallelType::TIDx);

      red_tv_rf->axis(1)->parallelize(ParallelType::TIDy);
      red_tv->axis(1)->parallelize(ParallelType::TIDy);
  // Do a cross-warp reduction per block
    } else {
      if (rparams.cross_block_) {
        red_tv->split(-1, rparams.block_dim_x_);
        // Split up rFactor to reduce across warps
        red_tv->split(-2, rparams.grid_dim_y_);
        red_tv->split(-3, rparams.block_dim_y_);

        auto red_tv_rf = red_tv->rFactor({-4});
        red_tv_rf->computeAt(red_tv, 1);

        red_tv->axis(0)->parallelize(ParallelType::BIDx);

        // Cross-block reduction binding
        red_tv_rf->axis(-3)->parallelize(ParallelType::BIDy);
        red_tv_rf->axis(-2)->parallelize(ParallelType::TIDy);
        red_tv_rf->axis(-1)->parallelize(ParallelType::TIDx);

        red_tv->axis(-3)->parallelize(ParallelType::BIDy);
        red_tv->axis(-2)->parallelize(ParallelType::TIDy);
        red_tv->axis(-1)->parallelize(ParallelType::TIDx);

      } else {
        red_tv->split(-1, rparams.block_dim_x_);
        // Split up rFactor to reduce across warps
        red_tv->split(-2, rparams.block_dim_y_);

        auto red_tv_rf = red_tv->rFactor({-3});
        red_tv_rf->computeAt(red_tv, 1);

        red_tv->axis(0)->parallelize(ParallelType::BIDx);

        red_tv_rf->axis(-2)->parallelize(ParallelType::TIDy);
        red_tv_rf->axis(-1)->parallelize(ParallelType::TIDx);

        red_tv->axis(-2)->parallelize(ParallelType::TIDy);
        red_tv->axis(-1)->parallelize(ParallelType::TIDx);
      }
    }
  } else {
    if (rparams.block_dim_y_ > 1) {
      red_tv->split(-1, rparams.block_dim_x_);
      if(rparams.grid_dim_y_ > 1)
        red_tv->split(0,rparams.grid_dim_y_);
      red_tv->split(0, rparams.block_dim_y_);
      auto red_tv_rf = red_tv->rFactor({0});
      red_tv_rf->axis(-1)->parallelize(ParallelType::TIDx);
      red_tv_rf->axis(-2)->parallelize(ParallelType::BIDx);
      if(rparams.grid_dim_y_ > 1) {
        red_tv_rf->axis(-3)->parallelize(ParallelType::BIDy);
        red_tv_rf->axis(-4)->parallelize(ParallelType::TIDy);
      } else {
        red_tv_rf->axis(-3)->parallelize(ParallelType::TIDy);
      }
      red_tv->axis(-1)->parallelize(ParallelType::TIDx);
      red_tv->axis(-2)->parallelize(ParallelType::BIDx);
      red_tv->axis(-3)->parallelize(ParallelType::TIDy);
      if(rparams.grid_dim_y_ > 1) {
        red_tv->axis(-3)->parallelize(ParallelType::BIDy);
        red_tv->axis(-4)->parallelize(ParallelType::TIDy);
      } else {
        red_tv->split(0, rparams.block_dim_x_);
        red_tv->split(2, rparams.block_dim_y_);
        auto red_tv_rf = red_tv->rFactor({2});

        // Bindings
        red_tv_rf->axis(1)->parallelize(ParallelType::TIDx);
        red_tv_rf->axis(0)->parallelize(ParallelType::BIDx);
        red_tv_rf->axis(-1)->parallelize(ParallelType::TIDy);

        red_tv->axis(1)->parallelize(ParallelType::TIDx);
        red_tv->axis(0)->parallelize(ParallelType::BIDx);
        red_tv->axis(-1)->parallelize(ParallelType::TIDy);
      }
    } else {
      red_tv->split(0, rparams.block_dim_x_);
      red_tv->axis(0)->parallelize(ParallelType::TIDx);
      red_tv->axis(1)->parallelize(ParallelType::BIDx);
    }
  }

  // Communicate Blocking for Kernel Launch
  fusion->setLaunchConfig(LaunchConfigType::TIDx, new Int(rparams.block_dim_x_));
  fusion->setLaunchConfig(LaunchConfigType::TIDy, new Int(rparams.block_dim_y_));
  fusion->setLaunchConfig(LaunchConfigType::TIDz, new Int(1));
  fusion->setLaunchConfig(LaunchConfigType::BIDx, new Int(rparams.grid_dim_x_));
  fusion->setLaunchConfig(LaunchConfigType::BIDy, new Int(rparams.grid_dim_y_));
  fusion->setLaunchConfig(LaunchConfigType::BIDz, new Int(1));
  fusion->setLaunchConfig(LaunchConfigType::SharedMemory, new Int(0));

  return rparams;
}



} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch

#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/ArrayRef.h>

#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>
#include <torch/csrc/jit/codegen/cuda/kernel_arg.h>
#include <torch/csrc/jit/codegen/cuda/kernel_resource_strings.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/parser.h>

#include <torch/csrc/jit/resource_guard.h>
#include <fstream>
#include <iostream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

constexpr auto kCgNamespace = "CudaCodeGen";
constexpr auto kKernelName = "kernel";

namespace {

// See NOTE [ USE OF NVRTC AND DRIVER API ]
const at::cuda::NVRTC& nvrtc() {
  return at::globalContext().getNVRTC();
}

int ceilDiv(const int a, const int b) {
  return (a + b - 1) / b;
}

struct KernelArgumentHolder {
 private:
  std::vector<ArgAbstract*> arguments;
  std::vector<void*> void_ptrs;
  bool changed = true;

 public:
  virtual ~KernelArgumentHolder() {
    for (auto arg : arguments)
      delete arg;
  }

  // Push a tensor to the arguments
  void push(
      const at::Tensor& val,
      c10::optional<at::IntArrayRef> broadcasted_size = c10::nullopt) {
    changed = true;
    ExtractSizeStride ess(val, std::move(broadcasted_size));
    int nDims = ess.sizes.size();

    c10::ScalarType dtype = val.scalar_type();
    TensorArgAbstract* tensor_arg = getTensorArg(dtype, nDims);
    tensor_arg->setPointer(val.data_ptr());
    for (int i = 0; i < nDims; i++) {
      tensor_arg->setSize(i, ess.sizes[i]);
      tensor_arg->setStride(i, ess.strides[i]);
    }
    arguments.push_back(tensor_arg);
  }

  // Push a scalar or integer to the arguments
  void push(const IValue& val) {
    changed = true;
    TORCH_INTERNAL_ASSERT(
        val.isScalar(),
        "Tried to push an arg to run in a fused kernel, expected a scalar but got, ",
        val);
    switch (val.toScalar().type()) {
      case (c10::ScalarType::Double):
        arguments.push_back(new FloatArg((float)val.toDouble()));
        return;
      case (c10::ScalarType::Long):
        arguments.push_back(new IntArg((int)val.toInt()));
        return;
      default:
        TORCH_INTERNAL_ASSERT(
            false,
            " Tried to create argument to send to a fused kernel, but got an unexpected type.");
    }
    TORCH_INTERNAL_ASSERT(
        false,
        " Tried to create argument to send to a fused kernel, but got a non-scalar type.");
  }

  void push(const uint64_t& val) {
    arguments.push_back(new ULongArg(val));
  }

  // Create buffer, flatten arguments into it, align by 8 Bytes, return pointers
  // in the buffer
  void** getBuffer() {
    if (changed) {
      void_ptrs = std::vector<void*>(arguments.size(), nullptr);
      for (decltype(arguments.size()) i{0}; i < arguments.size(); i++)
        void_ptrs[i] = static_cast<void*>(arguments[i]->arg());
      changed = false;
    }
    return void_ptrs.data();
  }
};

std::pair<std::string, std::string> codeGeneration(Fusion* fusion) {
  std::stringstream str_stream;
  str_stream << "namespace " << kCgNamespace << " {\n"
             << code_template_tensor_struct << "\n"
             << code_fp16_support << "\n"
             << code_random_number_gen << "\n"
             << code_helper_funcs << "\n"
             << code_template_block_reduction << "\n"
             << code_template_grid_reduction << "\n"
             << code_template_block_broadcast << "\n";
  std::stringstream cdg;
  GPULower gpulw(fusion);
  gpulw.printKernel(str_stream, kKernelName);
  str_stream << "\n} // namespace";

  std::string func_name = std::string(kCgNamespace) + "::" + kKernelName;
  return std::make_pair(func_name, str_stream.str());
}

bool validateKernelArgTensor(
    const at::Tensor& arg,
    const Val* param,
    int device_index,
    std::stringstream& msg) {
  // Arg is a tensor. Param must be a tensor too.
  if (*param->getValType() != ValType::TensorView) {
    msg << "Argument is a tensor, but the parameter is not.";
    return false;
  }

  // Check the rank of the tensors.
  size_t arg_dim = arg.dim();
  // Note: This requires current Fusion to be active.
  size_t param_dim = TensorDomain::noReductions(
                         static_cast<const TensorView*>(param)->getRootDomain())
                         .size();
  // see [Note - broadcast support in integration]
  // Because of broadcasting support handled in integration, we relax the rank
  // check as necessary.
  if (arg_dim > param_dim) {
    msg << "Argument tensor's rank is " << arg_dim << ", but the parameter is "
        << param_dim;
    return false;
  }

  if (arg.device().index() != device_index) {
    msg << "Argument is on device that is not compiled for";
    return false;
  }
  // Check element type
  at::ScalarType arg_data_type = arg.scalar_type();
  DataType param_data_type = *param->getDataType();
  bool match = false;
  switch (arg_data_type) {
    case at::ScalarType::Half:
      match = param_data_type == DataType::Half;
      break;
    case at::ScalarType::Float:
      match = param_data_type == DataType::Float;
      break;
    case at::ScalarType::Bool:
      match = param_data_type == DataType::Bool;
      break;
    default:
      msg << "Argument element type, " << arg_data_type
          << ", is not supported.";
      return false;
  }
  if (!match)
    msg << "Argument element type is " << arg_data_type
        << ", but the parameter is " << param_data_type;
  return match;
}

bool validateKernelArgScalar(
    const c10::TypePtr& arg_type,
    const Val* param,
    std::stringstream& msg) {
  if (!param->isScalar()) {
    msg << "Argument is a scalar, but the parameter is not.";
    return false;
  }
  DataType param_type = *param->getDataType();
  bool match = false;
  switch (arg_type->kind()) {
    case c10::TypeKind::IntType:
      match = param_type == DataType::Int;
      break;
    case c10::TypeKind::FloatType:
      match = param_type == DataType::Float;
      break;
    case c10::TypeKind::BoolType:
      match = param_type == DataType::Bool;
      break;
    default:
      match = false;
  }
  if (!match) {
    msg << "Argument type is " << *arg_type << ", but the parameter is "
        << param_type;
  }
  return match;
}

bool validateKernelArg(
    const c10::IValue& arg,
    const Val* param,
    int device_index,
    std::stringstream& msg) {
  if (arg.type()->kind() != c10::TypeKind::TensorType) {
    return validateKernelArgScalar(arg.type(), param, msg);
  } else {
    return validateKernelArgTensor(arg.toTensor(), param, device_index, msg);
  }
}

void validateKernelArgs(
    CudaKernel* entry,
    const at::ArrayRef<IValue>& inputs,
    const std::vector<at::Tensor>& outputs) {
  // This is necessary as we were traversing the fusion graph later in the check
  FusionGuard fg(entry);
  // Check inputs
  TORCH_INTERNAL_ASSERT(
      inputs.size() == entry->fusion()->inputs().size(),
      "Wrong number of kernel inputs.");
  for (size_t i = 0; i < inputs.size(); ++i) {
    const IValue& arg = inputs[i];
    const Val* param = entry->fusion()->inputs()[i];
    std::stringstream msg;
    TORCH_INTERNAL_ASSERT(
        validateKernelArg(arg, param, entry->device(), msg),
        "Input argument at position ",
        i,
        " is invalid; ",
        msg.str());
  }

  TORCH_INTERNAL_ASSERT(
      entry->fusion()->outputs().size() != 0,
      "Kernel should have at least one output tensor.");

  TORCH_INTERNAL_ASSERT(
      outputs.size() == entry->fusion()->outputs().size(),
      "Wrong number of kernel outputs.");
  for (size_t i = 0; i < outputs.size(); ++i) {
    const at::Tensor& arg = outputs[i];
    const Val* param = entry->fusion()->outputs()[i];
    std::stringstream msg;
    TORCH_INTERNAL_ASSERT(
        validateKernelArgTensor(arg, param, entry->device(), msg),
        "Output argument at position ",
        i,
        " is invalid; ",
        msg.str());
  }
}

size_t size(const dim3& d) {
  return (size_t)d.x * (size_t)d.y * (size_t)d.z;
}

dim3 dimensionOfReductionBlock(
    const dim3& block_dim,
    bool x_thread,
    bool y_thread,
    bool z_thread) {
  return dim3{x_thread ? block_dim.x : 1,
              y_thread ? block_dim.y : 1,
              z_thread ? block_dim.z : 1};
}

int sizeOfReductionBlock(
    const dim3& block_dim,
    bool x_thread,
    bool y_thread,
    bool z_thread) {
  return size(
      dimensionOfReductionBlock(block_dim, x_thread, y_thread, z_thread));
}

// Returns the total number of reduction segments.
size_t numberOfReductionSegments(
    const dim3& grid_dim,
    bool x_block,
    bool y_block,
    bool z_block) {
  return (x_block ? 1 : grid_dim.x) * (y_block ? 1 : grid_dim.y) *
      (z_block ? 1 : grid_dim.z);
}

std::array<size_t, 2> gridReductionTempBufferSizes(
    CudaKernel* entry,
    const dim3& grid_dim,
    const dim3& block_dim) {
  size_t buffer_size = 0;
  size_t sync_flag_size = 0;
  for (auto expr : entry->fusion()->exprs(true)) {
    if (expr->getExprType() != ExprType::ReductionOp)
      continue;
    ReductionOp* rop = static_cast<ReductionOp*>(expr);
    auto domains = rop->getParallelReductionDomains();
    bool x_block = domains.find(ParallelType::BIDx) != domains.end();
    bool y_block = domains.find(ParallelType::BIDy) != domains.end();
    bool z_block = domains.find(ParallelType::BIDz) != domains.end();
    // No buffer needed unless it's a grid reduction
    if (!x_block && !y_block && !z_block)
      continue;
    // Assumption here is that reduction along the block-parallel
    // domains is done prior to this grid reduction, so those domains
    // do not need to participate in the grid reductions
    bool x_thread = domains.find(ParallelType::TIDx) == domains.end();
    bool y_thread = domains.find(ParallelType::TIDy) == domains.end();
    bool z_thread = domains.find(ParallelType::TIDz) == domains.end();
    auto rb_size =
        sizeOfReductionBlock(block_dim, x_thread, y_thread, z_thread);
    auto num_blocks = size(grid_dim);
    auto element_size = dataTypeSize(*(rop->out()->getDataType()));
    auto required_temp_buffer_size = num_blocks * rb_size * element_size;
    buffer_size = std::max(buffer_size, required_temp_buffer_size);
    auto flag_size = sizeof(unsigned) *
        numberOfReductionSegments(grid_dim, x_block, y_block, z_block);
    sync_flag_size = std::max(sync_flag_size, flag_size);
  }
  return {{buffer_size, sync_flag_size}};
}

} // namespace

void compileKernel(CudaKernel* entry) {
  // generating cuda code;
  std::string code;
  std::string func_name;
  std::tie(func_name, code) = codeGeneration(entry->fusion());

  static int32_t compiled_kernel_id = 0;
  // We increment the id here instead of at the end of the function to avoid
  // error during jit-compilation that would make debug message confusing.
  compiled_kernel_id++;
  const char* debug_env = getenv("PYTORCH_CUDA_FUSER_DEBUG");
  if (debug_env && atoi(debug_env)) {
    std::cout << "\n==== codegen output for kernel: " << compiled_kernel_id
              << " ====" << std::endl
              << code << std::endl
              << "====================================" << std::endl;
  }

  // vvv NVRTC COMPILATION vvv

  // lazily construct context if non-existing yet;
  CUcontext pctx = nullptr;
  AT_CUDA_DRIVER_CHECK(nvrtc().cuCtxGetCurrent(&pctx));
  if (!pctx) {
    std::unique_lock<std::mutex> cudaFreeMutexLock(
        *(c10::cuda::CUDACachingAllocator::getFreeMutex()));
    cudaFree(nullptr);
  }

  // set device for the operation;
  at::cuda::set_device(entry->device());

  const auto prop = at::cuda::getCurrentDeviceProperties();
  int nvrtc_major, nvrtc_minor;
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcVersion(&nvrtc_major, &nvrtc_minor));

  // Short-circuits if NVRTC version too low
  TORCH_INTERNAL_ASSERT(nvrtc_major >= 6);
  // Major and minor is determined by device properties and
  // possibly "downcompiled" to a lower (compatible) compute architecture
  // based on the NVRTC version
  int major, minor;
  major = prop->major;
  minor = prop->minor;
  nvrtcProgram program;
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcCreateProgram(
      &program, code.c_str(), nullptr, 0, nullptr, nullptr));
  ResourceGuard holdProgram(
      [&] { AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcDestroyProgram(&program)); });

  const std::string compute = "--gpu-architecture=compute_" +
      std::to_string(major) + std::to_string(minor);
  const std::vector<const char*> args = {
      "--std=c++14", compute.c_str(), "-default-device"};

  nvrtc().nvrtcAddNameExpression(program, func_name.c_str());
  const auto result =
      nvrtc().nvrtcCompileProgram(program, args.size(), args.data());
  if (result != NVRTC_SUCCESS) {
    size_t logsize;
    nvrtc().nvrtcGetProgramLogSize(program, &logsize);
    std::vector<char> log(logsize);
    nvrtc().nvrtcGetProgramLog(program, log.data());

    TORCH_INTERNAL_ASSERT(
        false, code.c_str(), "\nCUDA NVRTC compile error: ", log.data());
  }
  const char* lowered_kernel_name;
  nvrtc().nvrtcGetLoweredName(program, func_name.c_str(), &lowered_kernel_name);

  AT_CUDA_NVRTC_CHECK(result);
  size_t ptx_size;
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcGetPTXSize(program, &ptx_size));
  std::vector<char> ptx;
  ptx.resize(ptx_size);
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcGetPTX(program, ptx.data()));

  // TODO: We do go through different code path, should investigate whether this
  // has an impact on generated binary.
  const char* prefix_env = getenv("PYTORCH_CUDA_FUSER_CUBIN");
  if (prefix_env) {
    // Output ptx file
    std::stringstream ptx_file_name;
    ptx_file_name << prefix_env << "_" << compiled_kernel_id << ".ptx";
    std::ofstream myPtxFile(ptx_file_name.str().c_str(), std::ios::out);
    if (myPtxFile.is_open()) {
      myPtxFile.write(ptx.data(), ptx.size());
      myPtxFile.close();
    }

    CUlinkState linkState;

    AT_CUDA_DRIVER_CHECK(nvrtc().cuLinkCreate(0, nullptr, nullptr, &linkState));
    AT_CUDA_DRIVER_CHECK(nvrtc().cuLinkAddData(
        linkState,
        CU_JIT_INPUT_PTX,
        ptx.data(),
        ptx_size,
        "compiling PTX",
        0,
        nullptr,
        nullptr));
    size_t cubinSize;
    void* cubin;
    AT_CUDA_DRIVER_CHECK(nvrtc().cuLinkComplete(linkState, &cubin, &cubinSize));

    // Output binary file
    std::stringstream cubin_file_name;
    cubin_file_name << prefix_env << "_" << compiled_kernel_id << ".cubin";
    std::ofstream myCubinFile(
        cubin_file_name.str().c_str(), std::ios::out | std::ios::binary);
    if (myCubinFile.is_open()) {
      myCubinFile.write(static_cast<const char*>(cubin), cubinSize);
      myCubinFile.close();
    }

    // load compiled cubin
    AT_CUDA_DRIVER_CHECK(nvrtc().cuModuleLoadData(entry->module(), cubin));
  } else {
    // load ptx directly
    AT_CUDA_DRIVER_CHECK(nvrtc().cuModuleLoadData(entry->module(), ptx.data()));
  }
  AT_CUDA_DRIVER_CHECK(nvrtc().cuModuleGetFunction(
      entry->function(), *entry->module(), lowered_kernel_name));
}

void runKernel(
    CudaKernel* entry,
    const at::ArrayRef<IValue> inputs,
    const std::vector<at::Tensor>& outputs,
    const c10::optional<at::IntArrayRef>& broadcasted_size) {
  validateKernelArgs(entry, inputs, outputs);

  const auto prior_device = at::cuda::current_device();
  at::cuda::set_device(entry->device());
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_INTERNAL_ASSERT(!outputs.empty(), "No outputs set for test kernel.");
  const size_t numel = outputs[0].numel();

  KernelArgumentHolder kernel_args;

  // Naive I/O setup, I'm ignoring all the potential transformation (i.e. I/O
  // allocated here from the subgraph could be, and very likely are, different
  // from I/O expected by the generated CUDA kernel.
  for (auto& input : inputs) {
    if (input.isTensor()) {
      kernel_args.push(input.toTensor(), broadcasted_size);
    } else {
      kernel_args.push(input);
    }
  }

  for (auto& output : outputs) {
    kernel_args.push(output);
  }

  Fusion* fusion = entry->fusion();
  FusionGuard fg(fusion);
  EvaluationContext eval_context(fusion);
  for (int i = 0; i < (int)inputs.size(); i++) {
    if (inputs[i].isTensor()) {
      ExtractSizeStride ess(inputs[i].toTensor(), broadcasted_size);
      int nDims = ess.sizes.size();
      TensorView* tv = fusion->inputs()[i]->as<TensorView>();
      for (int j = 0; j < nDims; j++) {
        eval_context.bind(tv->getRootDomain()[j]->extent(), ess.sizes[j]);
      }
    }
  }

  auto expr_eval_fn = [&](LaunchConfigType type) {
    const auto val = ExpressionEvaluator::evaluate(
        fusion->getLaunchConfig(type), &eval_context);
    TORCH_CHECK(
        val.has_value(), "scheduler didn't bind launch configs properly");
    return val.value();
  };

  const int nBlocks_x = expr_eval_fn(LaunchConfigType::BIDx);
  const int nBlocks_y = expr_eval_fn(LaunchConfigType::BIDy);
  const int nBlocks_z = expr_eval_fn(LaunchConfigType::BIDz);
  const auto nThreadx = expr_eval_fn(LaunchConfigType::TIDx);
  const auto nThready = expr_eval_fn(LaunchConfigType::TIDy);
  const auto nThreadz = expr_eval_fn(LaunchConfigType::TIDz);
  const auto shared_memory = expr_eval_fn(LaunchConfigType::SharedMemory);

  dim3 grid_dim(nBlocks_x, nBlocks_y, nBlocks_z);
  dim3 block_dim(nThreadx, nThready, nThreadz);

  // TODO: this probably won't work for us.
  if (entry->hasRNG()) {
    std::pair<uint64_t, uint64_t> philox_engine_inputs;
    const auto rand_offset =
        4 * (std::ceil(numel / (4.0 * 128 * nBlocks_x)) + 1);
    auto gen = at::cuda::detail::getDefaultCUDAGenerator();
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(gen.mutex());
      philox_engine_inputs =
          at::check_generator<at::CUDAGeneratorImpl>(gen)->philox_engine_inputs(
              rand_offset);
    }
    kernel_args.push(philox_engine_inputs.first);
    kernel_args.push(philox_engine_inputs.second);
  }

  dim3 grid_dim(nBlocks_x, nBlocks_y, nBlocks_z);
  dim3 block_dim(nThreadx, nThready, nThreadz);
  // When the kernel has global reductions, the kernel needs two
  // additional temporary buffers, one for intermediate results and
  // another for synchronization among thread blocks.
  if (entry->fusion()->hasGridReduction()) {
    auto temp_buf_type = at::kFloat;
    auto temp_buf_sizes =
        gridReductionTempBufferSizes(entry, grid_dim, block_dim);
    auto options =
        at::TensorOptions().dtype(temp_buf_type).device(at::kCUDA, 0);
    at::Tensor reduction_work_buffer = at::empty(
        {(long)(temp_buf_sizes[0] / c10::elementSize(temp_buf_type))}, options);
    kernel_args.push(reduction_work_buffer);
    at::Tensor sync_flags = at::zeros(
        {(long)(temp_buf_sizes[1] / c10::elementSize(temp_buf_type))}, options);
    kernel_args.push(sync_flags);
  }

  // launch kernel;
  AT_CUDA_DRIVER_CHECK(nvrtc().cuLaunchKernel(
      *entry->function(),
      nBlocks_x,
      nBlocks_y,
      nBlocks_z,
      nThreadx,
      nThready,
      nThreadz,
      shared_memory,
      stream,
      kernel_args.getBuffer(),
      nullptr));

  // Resets device (see at::DeviceGuard notes above)
  at::cuda::set_device(prior_device);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch

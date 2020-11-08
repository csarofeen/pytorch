#include <torch/csrc/jit/codegen/cuda/parser.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>

#include <torch/csrc/jit/frontend/function_schema_parser.h>
#include <torch/csrc/jit/ir/constants.h>

#include <unordered_map>
#include <utility>

namespace torch {
namespace jit {

typedef Value JitValue;
typedef Node JitOp;

namespace fuser {
namespace cuda {

constexpr auto kNumUnaryOps = 31;
constexpr auto kNumBinaryOps = 24;
constexpr auto kNumBinaryOpsWithAlpha = 4;
constexpr auto kNumLerpOps = 2;

namespace {

typedef Val* CgValue;
typedef Expr* CgOp;

// TODO: add a mutex to make it thread safe.
class IrParser {
  struct ValueEntry {
    ValueEntry() = default;
    ValueEntry(CgValue val) : val_(val) {}

    CgValue val() {
      TORCH_INTERNAL_ASSERT(isValue() && !isContainer());
      return val_;
    }

    void setVal(CgValue val) {
      TORCH_INTERNAL_ASSERT(isValue() && !isContainer());
      val_ = val;
    }

    const std::vector<CgValue>& vec() {
      TORCH_INTERNAL_ASSERT(!isValue() && isContainer());
      return vec_;
    }

    bool isValue() {
      return val_ != nullptr;
    }

    bool isContainer() {
      return !vec_.empty();
    }

    CgValue val_ = nullptr;
    std::vector<CgValue> vec_;
  };

  typedef std::unordered_map<size_t, ValueEntry> ValMap;
  typedef void (*ParseFuncPtr)(const Node*, ValMap&);
  typedef bool (*MergeQueryFuncPtr)(const Node*);

  class RegistrationEntry {
   public:
    RegistrationEntry(ParseFuncPtr parse_f, MergeQueryFuncPtr merge_f = nullptr)
        : parse_f_(parse_f), merge_f_(merge_f) {}

    void parse(const Node* node, ValMap& values) {
      parse_f_(node, values);
    }

    bool is_compatible(const Node* node) {
      if (merge_f_ == nullptr) {
        return true;
      }
      return merge_f_(node);
    }

   private:
    ParseFuncPtr parse_f_;
    MergeQueryFuncPtr merge_f_;
  };

 public:
  IrParser(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {
    if (init_registry_) {
      registerJitOperator();
      init_registry_ = false;
    }
  }

  // Fuses pointwise ops with loop unrolling (factor = 4).
  std::unique_ptr<Fusion> parse() {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    auto block = graph_->block();

    // register all inputs;
    for (auto val : block->inputs()) {
      if (registerValue(val)) {
        fusion->addInput(value_map_[val->unique()].val());

        auto opt_dtype = value_map_[val->unique()].val()->getDataType();
        // computation promotion, we cast fp16 inputs to fp32 and use promoted
        // type in the computation.
        if (opt_dtype.has_value() && opt_dtype.value() == DataType::Half) {
          Val* promoted_val = castOp(DataType::Float, value_map_[val->unique()].val());
          value_map_[val->unique()].setVal(promoted_val);
        }
      } else if (registerList(val)) {
        // we register IntList
      }
    }

    // compose nodes in topo order;
    for (const JitOp* node : block->nodes()) {
      processJitNode(node);
    }

    // mark output;
    for (auto jit_output : block->outputs()) {
      TensorView* out = value_map_[jit_output->unique()].val()->as<TensorView>();
      // demote output dtype to be match PyTorch JIT graph.
      auto tensor_type = jit_output->type()->cast<TensorType>();
      TORCH_INTERNAL_ASSERT(
          tensor_type, "output of fusion group is not TensorType.");
      if (tensor_type->scalarType() == at::ScalarType::Half) {
        // No need to update value_map_ after this point.
        out = castOp(DataType::Half, out)->as<TensorView>();
      }
      fusion->addOutput(out);
    }
    return fusion;
  }

  static bool canParseNode(const Node* node) {
    if (init_registry_) {
      // TODO: mutex this guy;
      registerJitOperator();
      init_registry_ = false;
    }

    // match signature.
    auto iter = jit_operator_registry_.find(node->kind());
    if (iter == jit_operator_registry_.end()) {
      return false;
    }
    for (auto& pair_op_func : iter->second) {
      if (node->matches(pair_op_func.first->schema())) {
        return pair_op_func.second.is_compatible(node);
      }
    }
    return false;
  }

  static bool isReductionNode(const Node* node) {
    if (init_registry_) {
      // TODO: mutex this guy;
      registerJitOperator();
      init_registry_ = false;
    }

    return jit_reduction_op_registry_.count(node->kind());
  }

  // TODO: is_reduction is too hacky here. we should categorize operation types
  //       based on their memory accessing pattern, which would affect fusion
  //       strategy and partition logic.
  static void registerParseRule(
      std::shared_ptr<Operator>& op,
      ParseFuncPtr parse_fn,
      MergeQueryFuncPtr merge_query_fn = nullptr,
      bool is_reduction = false) {
    jit_operator_registry_[Symbol::fromQualString(op->schema().name())]
        .emplace_back(
            std::piecewise_construct,
            std::forward_as_tuple(op),
            std::forward_as_tuple(parse_fn, merge_query_fn));
    if (is_reduction) {
      jit_reduction_op_registry_.emplace(
          Symbol::fromQualString(op->schema().name()));
    }
  }

 private:
  static void registerJitOperator() {
    // Register parse-function for each JIT operator;
    // This is a one-time look up, our hash registry indexes on the pointer in
    // OperatorRegistry.

    std::array<const char*, kNumBinaryOpsWithAlpha> BinaryOpWithAlpha = {
        "aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
        "aten::add(Tensor self, Scalar other, Scalar alpha) -> Tensor",
        "aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
        "aten::sub(Tensor self, Scalar other, Scalar alpha) -> Tensor"};
    for (auto signature : BinaryOpWithAlpha) {
      auto ptr_op = getOperatorForLiteral(signature);
      registerParseRule(
          ptr_op,
          [](const Node* node,
             ValMap& value_map) -> void {
            using BinaryOpWithAlphaType = Val* (*)(Val*, Val*, Val*);
            static std::unordered_map<
                Symbol,
                std::pair<BinaryOpType, BinaryOpWithAlphaType>>
                op_mapping(
                    {{aten::add,
                      std::make_pair(
                          BinaryOpType::Add,
                          static_cast<BinaryOpWithAlphaType>(&add_alpha))},
                     {aten::sub,
                      std::make_pair(
                          BinaryOpType::Sub,
                          static_cast<BinaryOpWithAlphaType>(&sub_alpha))}});
            // TODO: handle scaling factor when it's not constant 1;
            auto lhs = value_map[node->inputs()[0]->unique()].val();
            auto rhs = value_map[node->inputs()[1]->unique()].val();
            auto alpha = value_map[node->inputs()[2]->unique()].val();

            if (alpha->isOneInt()) {
              auto out = binaryOp(op_mapping[node->kind()].first, lhs, rhs);
              value_map.emplace(node->output()->unique(), out);
            } else {
              auto out = op_mapping[node->kind()].second(lhs, rhs, alpha);
              value_map.emplace(node->output()->unique(), out);
            }
          });
    }

    std::array<const char*, kNumBinaryOps> BinaryOp = {
        "aten::div(Tensor self, Tensor other) -> Tensor",
        "aten::div(Tensor self, Scalar other) -> Tensor",
        "aten::mul(Tensor self, Tensor other) -> Tensor",
        "aten::mul(Tensor self, Scalar other) -> Tensor",
        "aten::atan2(Tensor self, Tensor other) -> Tensor",
        "aten::max(Tensor self, Tensor other) -> Tensor",
        "aten::min(Tensor self, Tensor other) -> Tensor",
        "aten::pow(Tensor self, Tensor exponent) -> Tensor",
        "aten::pow(Tensor self, Scalar exponent) -> Tensor",
        "aten::pow(Scalar self, Tensor exponent) -> Tensor",
        "aten::remainder(Tensor self, Tensor other) -> Tensor",
        "aten::fmod(Tensor self, Tensor other) -> Tensor",
        "aten::eq(Tensor self, Tensor other) -> Tensor",
        "aten::eq(Tensor self, Scalar other) -> Tensor",
        "aten::ne(Tensor self, Tensor other) -> Tensor",
        "aten::ne(Tensor self, Scalar other) -> Tensor",
        "aten::ge(Tensor self, Tensor other) -> Tensor",
        "aten::ge(Tensor self, Scalar other) -> Tensor",
        "aten::gt(Tensor self, Tensor other) -> Tensor",
        "aten::gt(Tensor self, Scalar other) -> Tensor",
        "aten::le(Tensor self, Tensor other) -> Tensor",
        "aten::le(Tensor self, Scalar other) -> Tensor",
        "aten::lt(Tensor self, Tensor other) -> Tensor",
        "aten::lt(Tensor self, Scalar other) -> Tensor"};
    for (auto signature : BinaryOp) {
      auto ptr_op = getOperatorForLiteral(signature);
      registerParseRule(
          ptr_op,
          [](const Node* node,
             ValMap& value_map) -> void {
            static std::unordered_map<Symbol, BinaryOpType> op_mapping(
                {{aten::div, BinaryOpType::Div},
                 {aten::mul, BinaryOpType::Mul},
                 {aten::add, BinaryOpType::Add},
                 {aten::sub, BinaryOpType::Sub},
                 {aten::atan2, BinaryOpType::Atan2},
                 {aten::min, BinaryOpType::Min},
                 {aten::max, BinaryOpType::Max},
                 {aten::pow, BinaryOpType::Pow},
                 {aten::remainder, BinaryOpType::Remainder},
                 {aten::fmod, BinaryOpType::Fmod},
                 {aten::lt, BinaryOpType::LT},
                 {aten::le, BinaryOpType::LE},
                 {aten::gt, BinaryOpType::GT},
                 {aten::ge, BinaryOpType::GE},
                 {aten::ne, BinaryOpType::NE},
                 {aten::eq, BinaryOpType::Eq}});
            auto lhs = value_map[node->inputs()[0]->unique()].val();
            auto rhs = value_map[node->inputs()[1]->unique()].val();

            auto out = binaryOp(op_mapping[node->kind()], lhs, rhs);
            value_map.emplace(node->output()->unique(), out);
          });
    }

    // TODO: cast operations should be merged in.
    std::array<const char*, kNumUnaryOps> UnaryOp = {
        "aten::neg(Tensor self) -> Tensor",
        "aten::abs(Tensor self) -> Tensor",
        "aten::log(Tensor self) -> Tensor",
        "aten::log10(Tensor self) -> Tensor",
        "aten::log1p(Tensor self) -> Tensor",
        "aten::log2(Tensor self) -> Tensor",
        "aten::lgamma(Tensor self) -> Tensor",
        "aten::exp(Tensor self) -> Tensor",
        "aten::expm1(Tensor self) -> Tensor",
        "aten::erf(Tensor self) -> Tensor",
        "aten::erfc(Tensor self) -> Tensor",
        "aten::cos(Tensor self) -> Tensor",
        "aten::acos(Tensor self) -> Tensor",
        "aten::cosh(Tensor self) -> Tensor",
        "aten::sin(Tensor self) -> Tensor",
        "aten::asin(Tensor self) -> Tensor",
        "aten::sinh(Tensor self) -> Tensor",
        "aten::tan(Tensor self) -> Tensor",
        "aten::tanh(Tensor self) -> Tensor",
        "aten::atan(Tensor self) -> Tensor",
        "aten::sqrt(Tensor self) -> Tensor",
        "aten::rsqrt(Tensor self) -> Tensor",
        "aten::ceil(Tensor self) -> Tensor",
        "aten::floor(Tensor self) -> Tensor",
        "aten::round(Tensor self) -> Tensor",
        "aten::trunc(Tensor self) -> Tensor",
        "aten::frac(Tensor self) -> Tensor",
        "aten::reciprocal(Tensor self) -> Tensor",
        "aten::relu(Tensor self) -> Tensor",
        "aten::sigmoid(Tensor self) -> Tensor",
        "aten::gelu(Tensor self) -> Tensor",
    };
    for (auto signature : UnaryOp) {
      auto ptr_op = getOperatorForLiteral(signature);
      registerParseRule(
          ptr_op,
          [](const Node* node,
             ValMap& value_map) -> void {
            static std::unordered_map<Symbol, UnaryOpType> op_mapping({
                {aten::neg, UnaryOpType::Neg},
                {aten::abs, UnaryOpType::Abs},
                {aten::log, UnaryOpType::Log},
                {aten::log10, UnaryOpType::Log10},
                {aten::log1p, UnaryOpType::Log1p},
                {aten::log2, UnaryOpType::Log2},
                {aten::lgamma, UnaryOpType::Lgamma},
                {aten::exp, UnaryOpType::Exp},
                {aten::expm1, UnaryOpType::Expm1},
                {aten::erf, UnaryOpType::Erf},
                {aten::erfc, UnaryOpType::Erfc},
                {aten::cos, UnaryOpType::Cos},
                {aten::acos, UnaryOpType::Acos},
                {aten::cosh, UnaryOpType::Cosh},
                {aten::sin, UnaryOpType::Sin},
                {aten::asin, UnaryOpType::Asin},
                {aten::sinh, UnaryOpType::Sinh},
                {aten::tan, UnaryOpType::Tan},
                {aten::tanh, UnaryOpType::Tanh},
                {aten::atan, UnaryOpType::Atan},
                {aten::sqrt, UnaryOpType::Sqrt},
                {aten::rsqrt, UnaryOpType::Rsqrt},
                {aten::ceil, UnaryOpType::Ceil},
                {aten::floor, UnaryOpType::Floor},
                {aten::round, UnaryOpType::Round},
                {aten::trunc, UnaryOpType::Trunc},
                {aten::frac, UnaryOpType::Frac},
                {aten::reciprocal, UnaryOpType::Reciprocal},
                {aten::relu, UnaryOpType::Relu},
                {aten::sigmoid, UnaryOpType::Sigmoid},
                {aten::gelu, UnaryOpType::Gelu},
            });
            auto operand = value_map[node->input()->unique()].val();

            auto out = unaryOp(op_mapping[node->kind()], operand);
            value_map.emplace(node->output()->unique(), out);
          });
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::rand_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor");
      registerParseRule(
          ptr_op,
          [](const Node* node,
             ValMap& value_map) -> void {
            auto operand = value_map[node->inputs()[0]->unique()].val();

            auto out = unaryOp(UnaryOpType::RandLike, operand);
            value_map.emplace(node->output()->unique(), out);
          });
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor");
      registerParseRule(
          ptr_op,
          [](const Node* node,
             ValMap& value_map) -> void {
            auto operand = value_map[node->inputs()[0]->unique()].val();
            auto th = value_map[node->inputs()[1]->unique()].val();
            auto value = value_map[node->inputs()[2]->unique()].val();

            auto out = threshold(operand, th, value);
            value_map.emplace(node->output()->unique(), out);
          });
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::clamp(Tensor self, Scalar? min, Scalar? max) -> Tensor");
      registerParseRule(
          ptr_op,
          [](const Node* node,
             ValMap& value_map) -> void {
            auto operand = value_map[node->inputs()[0]->unique()].val();
            // TODO: we need to get a proper lower bound per dtype in operand.
            auto low = value_map.count(node->inputs()[1]->unique()) != 0
                ? value_map[node->inputs()[1]->unique()].val()
                : new Float(std::numeric_limits<float>::min());
            auto high = value_map.count(node->inputs()[2]->unique()) != 0
                ? value_map[node->inputs()[2]->unique()].val()
                : new Float(std::numeric_limits<float>::max());

            auto out = clamp(operand, low, high);
            value_map.emplace(node->output()->unique(), out);
          });
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::where(Tensor condition, Tensor self, Tensor other) -> Tensor");
      registerParseRule(
          ptr_op,
          [](const Node* node,
             ValMap& value_map) -> void {
            auto condition = value_map[node->inputs()[0]->unique()].val();
            auto x = value_map[node->inputs()[1]->unique()].val();
            auto y = value_map[node->inputs()[2]->unique()].val();

            auto out = where(condition, x, y);
            value_map.emplace(node->output()->unique(), out);
          });
    }

    {
      std::array<const char*, kNumLerpOps> LerpOp = {
          "aten::lerp(Tensor self, Tensor end, Scalar weight) -> Tensor",
          "aten::lerp(Tensor self, Tensor end, Tensor weight) -> Tensor"};
      for (auto signature : LerpOp) {
        auto ptr_op = getOperatorForLiteral(signature);
        registerParseRule(
            ptr_op,
            [](const Node* node,
               ValMap& value_map) -> void {
              auto self = value_map[node->inputs()[0]->unique()].val();
              auto end = value_map[node->inputs()[1]->unique()].val();
              auto weight = value_map[node->inputs()[2]->unique()].val();

              auto out = lerp(self, end, weight);
              value_map.emplace(node->output()->unique(), out);
            });
      }
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor");
      registerParseRule(
          ptr_op,
          [](const Node* node,
             ValMap& value_map) -> void {
            auto self = value_map[node->inputs()[0]->unique()].val();
            auto tensor1 = value_map[node->inputs()[1]->unique()].val();
            auto tensor2 = value_map[node->inputs()[2]->unique()].val();
            auto value = value_map[node->inputs()[3]->unique()].val();

            auto out = addcmul(self, tensor1, tensor2, value);
            value_map.emplace(node->output()->unique(), out);
          });
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, int? dtype=None) -> (Tensor)");
      registerParseRule(
          ptr_op,
          [](const Node* node,
             ValMap& value_map) -> void {
            auto self = value_map[node->input(0)->unique()].val();
            auto dims_list = constant_as<c10::List<int64_t>>(node->input(1));
            TORCH_INTERNAL_ASSERT(
                dims_list.has_value(),
                "aten::sum cannot be fused with dynamic axes");
            std::vector<int> dims;
            for (const auto dim : dims_list->vec()) {
              dims.emplace_back(static_cast<int>(dim));
            }
            auto keepdim = constant_as<bool>(node->input(2));
            TORCH_INTERNAL_ASSERT(
                keepdim.has_value(),
                "aten::sum cannot be fused with dynamic keepdim");
            auto out = sum(self->as<TensorView>(), dims, keepdim.value());
            value_map.emplace(node->output()->unique(), out);
          },
          [](const Node* node) -> bool {
            // TODO: support cast of output types yet;
            if (!node->inputs()[3]->type()->isSubtypeOf(
                    static_cast<c10::TypePtr>(NoneType::get()))) {
              // We can only handle output as half and float;
              if (const auto opt_ivalue = toIValue(node->input(3))) {
                const auto scalar_type = opt_ivalue->toScalarType();
                if (scalar_type == at::ScalarType::Float ||
                    scalar_type == at::ScalarType::Half) {
                  return true;
                }
              }
              return false;
            }
            // we don't support dynamic reduction axes;
            if (node->inputs()[1]->node()->kind() != prim::Constant) {
              return false;
            }
            // we don't support dynamic keepdim yet;
            if (node->inputs()[2]->node()->kind() != prim::Constant) {
              return false;
            }
            return true;
          },
          true);
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::type_as(Tensor self, Tensor other) -> Tensor");
      registerParseRule(
          ptr_op,
          [](const Node* node,
             ValMap& value_map) -> void {
            auto self = value_map[node->inputs()[0]->unique()].val();

            // TODO: switch to PyTorch dtype as it's closer to truth.
            // For now, reality is that PyTorch IR profiling information could
            // be missing even with profiling executor, due to upstream
            // transformations between profiling runs to fusion pass.
            auto opt_dtype =
                value_map[node->inputs()[1]->unique()].val()->getDataType();
            TORCH_INTERNAL_ASSERT(opt_dtype.has_value());

            auto out = castOp(opt_dtype.value(), self);
            value_map.emplace(node->output()->unique(), out);
          });
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::sum_to_size(Tensor self, int[] size) -> Tensor");
      registerParseRule(
          ptr_op,
          [](const Node* node,
             ValMap& value_map) -> void {
            auto self = value_map[node->input(0)->unique()].val();
            auto dims_list = constant_as<c10::List<int64_t>>(node->input(1));
            TORCH_INTERNAL_ASSERT(
                dims_list.has_value(), "requires static reduce axes");
            std::vector<Int*> dims;
            for (const auto dim : dims_list->vec()) {
              dims.emplace_back(new Int(static_cast<int>(dim)));
            }
            auto out = sum_to(self->as<TensorView>(), dims);
            value_map.emplace(node->output()->unique(), out);
          },
          [](const Node* node) -> bool {
            // we only support static reduction sizes;
            return node->inputs()[1]->node()->kind() == prim::Constant;
          },
          true);
    }
  }

  void processJitNode(const JitOp* node) {
    if (node->kind() == prim::Constant) {
      // partition doesn't take constant node explicitly, but it does and copy
      // constant into subgraph. So we need to register constants in codegen IR;
      for (auto output : node->outputs()) {
        TORCH_INTERNAL_ASSERT(
            registerScalar(output),
            "registration of output failed at index ",
            output->offset(),
            " for node ",
            *node);
      }
    } else {
      auto iter = IrParser::jit_operator_registry_.find(node->kind());
      // make sure we have a parser for the op;
      TORCH_INTERNAL_ASSERT(
          iter != IrParser::jit_operator_registry_.end(),
          "CudaFusionGroup Parser doesn't handle operator kind(): ",
          node->kind().toDisplayString());
      for (auto& pair_op_func : iter->second) {
        if (node->matches(pair_op_func.first->schema())) {
          pair_op_func.second.parse(node, value_map_);
          return;
        }
      }
      TORCH_INTERNAL_ASSERT(
          false,
          "CudaFusionGroup Parser doesn't recognize operator overload:",
          canonicalSchemaString(node->schema()));
    }
  }

  bool registerValue(const JitValue* val) {
    return registerTensor(val) || registerScalar(val);
  }

  bool registerList(const JitValue* val) {
    TORCH_INTERNAL_ASSERT(false);
  }

  bool registerScalar(const JitValue* val) {
    if (val->type()->isSubtypeOf(static_cast<c10::TypePtr>(FloatType::get()))) {
      CgValue cg_val;
      if (auto ival = constant_as<float>(val)) {
        cg_val = new Float(ival.value());
      } else {
        cg_val = new Float();
      }
      value_map_.emplace(val->unique(), cg_val);
      return true;
    } else if (val->type()->isSubtypeOf(
                   static_cast<c10::TypePtr>(IntType::get()))) {
      CgValue cg_val;
      if (auto ival = constant_as<int>(val)) {
        cg_val = new Int(ival.value());
      } else {
        cg_val = new Int();
      }
      value_map_.emplace(val->unique(), cg_val);
      return true;
    } else if (val->type()->isSubtypeOf(
                   static_cast<c10::TypePtr>(BoolType::get()))) {
      CgValue cg_val;
      if (auto ival = constant_as<bool>(val)) {
        cg_val = new Bool(ival.value());
      } else {
        cg_val = new Bool();
      }
      value_map_.emplace(val->unique(), cg_val);
      return true;
    } else if (val->type()->isSubtypeOf(
                   static_cast<c10::TypePtr>(NoneType::get()))) {
      // TODO: should we consider adding support for NoneType;
      return true;
    } else if (val->type()->cast<ListType>()) {
      // For constant ListType, we don't need to register it as runtime input,
      // because graph conversion would inline the constants
      if (toIValue(val).has_value()) {
        return true;
      }
    }
    return false;
  }

  bool registerTensor(const JitValue* val) {
    CgValue cg_val;
    if (auto tensor_type = val->type()->cast<TensorType>()) {
      // TODO: make this a static function in Tensor class;
      // create tensor;
      cg_val = new TensorView(tensor_type);
      value_map_.emplace(val->unique(), cg_val);
      return true;
    }
    return false;
  }

  std::shared_ptr<Graph> graph_;

  // maps from JitValue::unique() to fusion Val;
  std::unordered_map<size_t, ValueEntry> value_map_;
  // parsing rule registry.
  static std::unordered_map<
      Symbol,
      std::vector<std::pair<std::shared_ptr<Operator>, RegistrationEntry>>>
      jit_operator_registry_;
  static std::unordered_set<Symbol> jit_reduction_op_registry_;
  static bool init_registry_;
};

std::unordered_map<
    Symbol,
    std::vector<
        std::pair<std::shared_ptr<Operator>, IrParser::RegistrationEntry>>>
    IrParser::jit_operator_registry_;
std::unordered_set<Symbol> IrParser::jit_reduction_op_registry_;
bool IrParser::init_registry_ = true;

ProfileIValueOp* insertProfileIValueOp(Node* node, size_t offset) {
  auto pn = new ProfileIValueOp(node->owningGraph(), nullptr);
  auto in_val = node->input(offset);
  pn->insertBefore(node);
  pn->addInput(in_val);
  auto pno = pn->addOutput();
  pno->setType(in_val->type());
  in_val->replaceAllUsesAfterNodeWith(pn, pno);
  pn->ty_(attr::profiled_type, in_val->type());
  return pn;
}

void profileIntList(ProfilingRecord* pr, Node* node, size_t offset) {
  auto pn = insertProfileIValueOp(node, offset);

  std::function<void(Stack&)> ivalue_profiler = [pr, pn] (Stack& stack) {
    std::lock_guard<std::mutex> lock(pr->mutex_);

    // TODO: we don't care about merging multiple profiling runs as we don't support it at all;
    int64_t frame_id = 0;
    pop(stack, frame_id);
    IValue value;
    pop(stack, value);
    TORCH_INTERNAL_ASSERT(value.isIntList(), "profiling seeing the wrong data type");
    // TODO: get a real attribute
    if (!pn->hasAttribute(attr::a)) {
      //pn->is_(attr::a, value.toIntList().vec());
      pn->is_(attr::a, value.toIntVector());
    } else {
      auto profiled_ints = pn->is(attr::a);
      auto input_ints = value.toIntList();
      TORCH_INTERNAL_ASSERT(profiled_ints.size() == input_ints.size() &&
          std::equal(profiled_ints.begin(), profiled_ints.end(), input_ints.begin()), "profiling ivalue doesn't support merge");
    }
    push(stack, value);
  };

  pn->setCallback(ivalue_profiler);
}

void profileBool(ProfilingRecord* pr, Node* node, size_t offset) {
  auto pn = insertProfileIValueOp(node, offset);

  std::function<void(Stack&)> ivalue_profiler = [pr, pn] (Stack& stack) {
    std::lock_guard<std::mutex> lock(pr->mutex_);

    // TODO: we don't care about merging multiple profiling runs as we don't support it at all;
    int64_t frame_id = 0;
    pop(stack, frame_id);
    IValue value;
    pop(stack, value);
    TORCH_INTERNAL_ASSERT(value.isBool(), "profiling seeing the wrong data type");
    // TODO: get a real attribute
    if (!pn->hasAttribute(attr::a)) {
      pn->i_(attr::a, value.toBool());
    } else {
      auto profiled_bool = pn->i(attr::a);
      auto input_bool = value.toBool();
      TORCH_INTERNAL_ASSERT(input_bool == profiled_bool, "profiling ivalue doesn't support merge");
    }
    push(stack, value);
  };

  pn->setCallback(ivalue_profiler);
}

} // namespace

bool hasReductionNode(const Block* block) {
  for (auto node : block->nodes()) {
    if (isReductionNode(node)) {
      return true;
    }
    for (auto block : node->blocks()) {
      if (hasReductionNode(block)) {
        return true;
      }
    }
  }
  return false;
}

bool isReductionNode(const Node* node) {
  return IrParser::isReductionNode(node);
}

bool isNodeParsible(const Node* node) {
  return IrParser::canParseNode(node);
}

// TODO: we should incorporate this to our parser as well;
bool insertProfileIValue(ProfilingRecord* pr, Node* node, size_t offset) {

  // is skip constant necessary?
  if (node->input(offset)->node()->kind() == prim::Constant) {
    return false;
  }

  // we should use `OperatorSet`
  static auto reduction_operator = Symbol::fromQualString(getOperatorForLiteral(
      "aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, int? dtype=None) -> (Tensor)")->schema().name());
  if (node->kind() == reduction_operator) {
    switch (offset) {
    case 1:
      profileIntList(pr, node, offset);
      break;
    case 2:
      profileBool(pr, node, offset);
      break;
    default:
      return false;
    }
    return true;
  }

  // we should use `OperatorSet`
  // static auto reduction_to_size_operator = getOperatorForLiteral(
  //     "aten::sum_to_size(Tensor self, int[] size) -> Tensor");
  // if (node->isMemberOf(reduction_to_size_operator) && offset == 1) {
  //   profileIntList(pr, node, offset);
  //   return true;
  // }

  return false;
}

std::unique_ptr<Fusion> parseJitIR(const std::shared_ptr<Graph>& graph) {
  FUSER_PERF_SCOPE("parseJitIR");

  IrParser parser(graph);
  return parser.parse();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch

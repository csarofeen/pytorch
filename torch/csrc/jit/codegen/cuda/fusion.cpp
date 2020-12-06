#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/codegen.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_cloner.h>
#include <torch/csrc/jit/codegen/cuda/ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>

#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

static thread_local Fusion* ACTIVE_FUSION = nullptr; // NOLINT

FusionGuard::FusionGuard(Fusion* fusion) {
  prev_fusion = ACTIVE_FUSION;
  ACTIVE_FUSION = fusion;
}

FusionGuard::~FusionGuard() {
  ACTIVE_FUSION = prev_fusion;
}

Fusion* FusionGuard::getCurFusion() {
  return ACTIVE_FUSION;
}

void swap(Fusion& a, Fusion& b) noexcept {
  FUSER_PERF_SCOPE("Fusion swap");

  using std::swap;

  // Swap the content
  swap(a.val_set_, b.val_set_);
  swap(a.expr_set_, b.expr_set_);
  swap(a.val_deque_, b.val_deque_);

  swap(a.val_type_name_map_, b.val_type_name_map_);
  swap(a.expr_name_counter_, b.expr_name_counter_);

  swap(a.inputs_, b.inputs_);
  swap(a.outputs_, b.outputs_);

  // Fixup the Statement::fusion_ links for a
  for (auto val : a.val_set_) {
    val->fusion_ = &a;
  }
  for (auto expr : a.expr_set_) {
    expr->fusion_ = &a;
  }

  // Fixup the Statement::fusion_ links for b
  for (auto val : b.val_set_) {
    val->fusion_ = &b;
  }
  for (auto expr : b.expr_set_) {
    expr->fusion_ = &b;
  }
}

Fusion::Fusion(const Fusion& other) {
  FUSER_PERF_SCOPE("Fusion copy");
  Fusion::copy(&other, this);
}

IrCloner Fusion::copy(const Fusion* from, Fusion* to) {
  to->clear();
  IrCloner ir_cloner(to);

  for (auto val : from->val_set_) {
    to->val_set_.insert(ir_cloner.clone(val));
  }

  for (auto val : from->val_deque_) {
    to->val_deque_.push_back(ir_cloner.clone(val));
  }

  for (auto old_expr : from->expr_set_) {
    auto new_expr = ir_cloner.clone(old_expr);
    to->expr_set_.insert(new_expr);

    // ir_cloner doesn't go through registerStmt, so we need to "Register Expr"
    // we would similarly need to do to val if there was in that pass that is
    // also not covered here.
    for (Val* input : new_expr->inputs()) {
      if (std::find(input->uses.begin(), input->uses.end(), new_expr) ==
          input->uses.end()) {
        input->uses.push_back(new_expr);
      }
    }
  }

  to->val_type_name_map_ = from->val_type_name_map_;
  to->expr_name_counter_ = from->expr_name_counter_;

  to->inputs_ = ir_cloner.clone(from->inputs_);
  to->outputs_ = ir_cloner.clone(from->outputs_);

  for (auto inp : to->inputs_) {
    inp->is_fusion_input = true;
  }
  for (auto out : to->outputs_) {
    out->is_fusion_output = true;
  }

  to->resetTvUses();
  return ir_cloner;
}

Fusion::Fusion(Fusion&& other) noexcept {
  FUSER_PERF_SCOPE("Fusion move");
  swap(*this, other);
}

Fusion& Fusion::operator=(const Fusion& other) {
  FUSER_PERF_SCOPE("Fusion copy assign");
  Fusion copy(other);
  clear();
  swap(*this, copy);
  return *this;
}

Fusion& Fusion::operator=(Fusion&& other) noexcept {
  FUSER_PERF_SCOPE("Fusion move assign");
  clear();
  swap(*this, other);
  return *this;
}

Fusion::~Fusion() {
  clear();
}

void Fusion::clear() noexcept {
  FUSER_PERF_SCOPE("Fusion clear");

  // Free the owned values
  for (auto ptr : val_set_) {
    delete ptr;
  }

  // Free the owned expressions
  for (auto ptr : expr_set_) {
    delete ptr;
  }

  val_set_.clear();
  val_deque_.clear();
  expr_set_.clear();

  for (auto& kv : val_type_name_map_) {
    kv.second = 0;
  }

  expr_name_counter_ = 0;

  inputs_.clear();
  outputs_.clear();
}

void Fusion::removeExpr(Expr* expr) {
  assertInFusion(expr, "Cannot remove expr ");
  // If we hit this error too frequently, we could lighten the restrictions so
  // that removing something that doesn't exist simply does nothing. For now,
  // we're going with the strictest model which errors.

  for (auto out : expr->outputs()) {
    out->origin = nullptr;
  }

  for (auto inp : expr->inputs()) {
    auto it = std::find(inp->uses.begin(), inp->uses.end(), expr);
    if (it != inp->uses.end()) {
      inp->uses.erase(it);
    }
  }

  expr_set_.erase(expr);

  delete expr;
}

void Fusion::removeVal(Val* val) {
  assertInFusion(val, "Cannot remove val ");

  TORCH_CHECK(
      !val->is_fusion_input,
      "Cannot remove val as it is an input of the fusion.");
  TORCH_CHECK(
      !val->is_fusion_output,
      "Cannot remove val as it is an output of the fusion.");

  Expr* orig = val->getOrigin();
  if (orig != nullptr)
    removeExpr(val->getOrigin());

  for (Expr* use : unordered_uses(val))
    removeExpr(use);

  val_set_.erase(val);

  for (auto it = val_deque_.begin(); it != val_deque_.end(); it++)
    if (*it == val) {
      val_deque_.erase(it);
      break;
    }

  delete val;
}

void Fusion::addInput(Val* input) {
  assertInFusion(input, "Cannot register input ");

  if (input->getValType().value() == ValType::TensorView) {
    auto tv = input->as<TensorView>();
    tv->setMemoryType(MemoryType::Global);
  }

  inputs_.push_back(input);
  input->is_fusion_input = true;

  resetTvUses();
}

void Fusion::addOutput(Val* output) {
  assertInFusion(output, "Cannot register output ");
  if (output->getValType().value() == ValType::TensorView) {
    auto tv = output->as<TensorView>();
    tv->setMemoryType(MemoryType::Global);
  }
  outputs_.push_back(output);
  output->is_fusion_output = true;

  resetTvUses();
}

void Fusion::removeInput(Val* input) {
  auto find_input = std::find(inputs_.begin(), inputs_.end(), input);
  if (find_input != inputs_.end()) {
    inputs_.erase(find_input);
  }
  input->is_fusion_input = false;
  resetTvUses();
}

void Fusion::removeOutput(Val* output) {
  auto find_output = std::find(outputs_.begin(), outputs_.end(), output);
  if (find_output != outputs_.end()) {
    outputs_.erase(find_output);
  }
  output->is_fusion_output = false;
  resetTvUses();
}

bool Fusion::inFusion(const Statement* stmt) const {
  bool in_fusion = stmt->fusion() == this;
  Statement* nonconst_stmt = const_cast<Statement*>(stmt); // NOLINT

  if (stmt->isExpr()) {
    in_fusion &= expr_set_.find(nonconst_stmt->as<Expr>()) != expr_set_.end();
  }
  if (stmt->isVal()) {
    in_fusion &= val_set_.find(nonconst_stmt->as<Val>()) != val_set_.end();
  }

  return in_fusion;
}

void Fusion::assertInFusion(const Statement* stmt, const std::string& msg)
    const {
  TORCH_CHECK(inFusion(stmt), msg, " it was not found in the active fusion.");
}

std::vector<Expr*> Fusion::exprs() {
  return ExprSort::getExprs(this);
}

std::unordered_set<Val*> Fusion::inputsOf(Val* val) {
  return InputsOf::output(this, val);
}

void Fusion::validateInputs() {
  std::unordered_set<Val*> all_inputs;
  for (Val* out : outputs()) {
    for (Val* input : inputsOf(out)) {
      all_inputs.insert(input);
    }
  }
  for (Val* input : all_inputs) {
    if (!input->isConstScalar()) {
      TORCH_CHECK(
          hasInput(input) || inFusion(input),
          "Could not figure out how ",
          input,
          " is generated, however it was not specified as an input.");
    }
  }
}

void Fusion::print() {
  FUSER_PERF_SCOPE("Fusion::print");

  FusionGuard fg(this);
  std::cout << "\n%kernel {\n";
  IrMathPrinter op_exprs(std::cout);
  op_exprs.handle(this);
  IrTransformPrinter t_exprs(std::cout);
  t_exprs.handle(this);
  std::cout << "}\n\n";
}

void Fusion::printKernel() {
  FUSER_PERF_SCOPE("Fusion::printKernel");
  std::cout << codegen::generateCudaKernel(GpuLower(this).kernel());
}

void Fusion::printMath(bool from_outputs_only) {
  FUSER_PERF_SCOPE("Fusion::printMath");

  FusionGuard fg(this);
  auto exprs_for_print = exprs();

  // If we want everything in the fusion, grab all values without uses to
  // traverse from.
  if (!from_outputs_only) {
    std::vector<Val*> leaf_vals;
    for (auto val : deterministic_vals()) {
      if (!val->uses.empty()) {
        leaf_vals.push_back(val);
      }
    }
    exprs_for_print = ExprSort::getExprs(this, leaf_vals);
  }

  std::cout << "\n%kernel_math {\n";
  for (auto expr : exprs_for_print) {
    std::cout << expr;
  }
  std::cout << "}\n\n";
}

void Fusion::printTransforms() {
  FUSER_PERF_SCOPE("Fusion::printTransforms");

  FusionGuard fg(this);
  IrTransformPrinter t_exprs(std::cout);
  t_exprs.handle(this);
}

StmtNameType Fusion::registerVal(Val* val) {
  if (val->fusion()) {
    if (val->fusion() != this) {
      TORCH_CHECK(false, val, " was not found in the active fusion.");
    }
    if (inFusion(val)) {
      return val->name();
    }
  }

  val_set_.emplace(val);
  val_deque_.push_back(val);
  return getValName(*(val->getValType()));
}

StmtNameType Fusion::registerExpr(Expr* expr) {
  if (expr->fusion()) {
    if (expr->fusion() != this) {
      TORCH_CHECK(false, expr, " was not found in the active fusion.");
    }
    if (inFusion(expr)) {
      return expr->name();
    }
  }

  for (Val* input : expr->inputs()) {
    assertInFusion(input, "Input to expr is invalid, ");
    if (std::find(input->uses.begin(), input->uses.end(), expr) ==
        input->uses.end()) {
      input->uses.push_back(expr);
    }
  }

  for (Val* output : expr->outputs()) {
    assertInFusion(output, "Output to expr is invalid, ");
    if (output->getOrigin() != nullptr) {
      removeExpr(output->getOrigin());
    }
    output->origin = expr;
  }

  expr_set_.emplace(expr);

  resetTvUses();
  return getExprName();
}

StmtNameType Fusion::registerStatement(Statement* stmt) {
  if (inFusion(stmt))
    return stmt->name();

  if (stmt->isVal()) {
    return registerVal(stmt->as<Val>());
  } else if (stmt->isExpr()) {
    return registerExpr(stmt->as<Expr>());
  }

  TORCH_INTERNAL_ASSERT(
      false,
      "Could not register statement as Fusion could not recognize its type.");
  return kInvalidStmName;
}

void Fusion::resetTvUses() {
  // getExprs only uses origin, so even if we've modified uses already to remove
  // dead exprs, this could reinsert them. getExprs is also boundeds by inputs
  // as registered inputs will return nullptr as their origin.
  auto all_tvs = ir_utils::filterByType<TensorView>(val_set_);
  auto used_exprs = ExprSort::getExprs(this);

  for (auto tv : all_tvs) {
    tv->uses.clear();
  }

  // Same as in register expr
  for (auto expr : used_exprs) {
    for (Val* input : expr->inputs()) {
      if (std::find(input->uses.begin(), input->uses.end(), expr) ==
          input->uses.end()) {
        input->uses.push_back(expr);
      }
    }
  }
}

const std::unordered_set<Val*>& Fusion::vals() const noexcept {
  return val_set_;
}

const std::deque<Val*>& Fusion::deterministic_vals() const noexcept {
  return val_deque_;
}

const std::unordered_set<Expr*>& Fusion::unordered_exprs() const noexcept {
  return expr_set_;
}

std::unordered_set<Expr*> Fusion::unordered_uses(Val* val) const {
  return std::unordered_set<Expr*>(val->uses.begin(), val->uses.end());
}

Expr* Fusion::origin(const Val* val) const {
  assertInFusion(val, "Cannot detect the origin of val, ");
  return val->getOrigin();
}

bool Fusion::hasInput(const Val* val) const {
  assertInFusion(val, "Cannot check if val is an input, ");
  return val->is_fusion_input;
}

bool Fusion::hasOutput(const Val* val) const {
  assertInFusion(val, "Cannot check if val is an output, ");
  return val->is_fusion_output;
}

StmtNameType Fusion::getValName(ValType vtype) {
  return val_type_name_map_[vtype]++;
}

StmtNameType Fusion::getExprName() {
  return expr_name_counter_++;
}

// Indicate to kernel to set itself up to generate random numbers
bool Fusion::isStochastic() {
  for (auto expr : exprs())
    if (expr->getExprType() == ExprType::UnaryOp)
      if (expr->as<UnaryOp>()->getUnaryOpType() == UnaryOpType::RandLike)
        return true;
  return false;
}

bool Fusion::hasReduction() {
  FUSER_PERF_SCOPE("Fusion::hasReduction");

  for (auto expr : exprs())
    for (auto out : expr->outputs())
      if (out->getValType() == ValType::TensorView)
        if (out->as<TensorView>()->hasReduction())
          return true;

  return false;
}

std::vector<Val*> Fusion::getTerminatingOutputs() {
  FUSER_PERF_SCOPE("getTerminatingOutputs");

  std::unordered_set<Val*> used_vals;

  const auto exprs = ExprSort::getExprs(
      this, std::vector<Val*>(outputs().begin(), outputs().end()));

  for (auto expr : exprs) {
    for (auto inp : expr->inputs())
      used_vals.emplace(inp);
  }

  std::vector<Val*> terminating_outputs;
  for (auto out : outputs()) {
    if (used_vals.find(out) != used_vals.end())
      continue;
    terminating_outputs.push_back(out);
  }
  return terminating_outputs;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch

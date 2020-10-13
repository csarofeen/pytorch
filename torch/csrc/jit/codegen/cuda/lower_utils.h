
#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

#include <bitset>
#include <map>

// Provides utilities for dealing with nested ForLoop and IfThenElse scopes

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace kir {
class ThreadPredicateMap;
}

using IterDomainMap = std::unordered_map<kir::IterDomain*, kir::IterDomain*>;

namespace scope_utils {

//! Returns the list of nesting loops starting at `scope`
//$$ needed?
std::vector<kir::ForLoop*> getLoops(kir::Expr* scope);

//! Insert expr in scope before ref
//!
//! \warning for kir::IfThenElse we implicitly insert in the "then" branch!
//!
void insertBefore(kir::Expr* scope, kir::Expr* ref, kir::Expr* expr);

} // namespace scope_utils

namespace ir_utils {

// Somtimes we want to temporarily view a tensorview with another tensordomain.
// This isn't a permanent transformation, but in indexing we want to index
// producers with a consumer set of indices, so we need to view the producer
// transformed like consumer while we index. This will set the tv with td for
// the life of this context guard.
class TVDomainGuard {
 private:
  TensorView* tv_;
  TensorDomain* prev_domain;

 public:
  explicit TVDomainGuard(TensorView* _tv, TensorDomain* td);

  ~TVDomainGuard();
};

// Return inputs of provided IterDomains that are IterDomains
std::vector<IterDomain*> iterDomainInputsOf(const std::vector<IterDomain*>&);

// Return inputs of provided IterDomains that are IterDomains, order as the
// second provided vector.
std::vector<IterDomain*> iterDomainInputsOfOrderedAs(
    const std::vector<IterDomain*>& of,
    const std::vector<IterDomain*>& order);

bool isTV(const Val* const);

bool isTVOp(const Expr*);

bool isTVOp(const kir::Expr* expr);

TensorView* getTVOutput(const Expr*);

bool isScalarOp(const Expr*);

bool hasChildScopes(const kir::Expr* expr);

// TODO(kir): remove
Expr* asExpr(Statement*);

// TODO(kir): Remove in favor of ->as<TensorView>()
TensorView* asTV(Val*);

// Represents mapping to bool from BIDx, BIDy, BIDz, TIDx, TIDy and TIDz.
class ParallelTypeBitmap {
 public:
  static constexpr int num_p_type = 6;
  ParallelTypeBitmap() = default;
  bool get(ParallelType pt) const;
  bool set(ParallelType pt, bool);
  ParallelTypeBitmap operator&=(const ParallelTypeBitmap& other);
  ParallelTypeBitmap operator|=(const ParallelTypeBitmap& other);
  ParallelTypeBitmap operator^=(const ParallelTypeBitmap& other);
  ParallelTypeBitmap operator~() const;
  bool none() const;
  bool any() const;
  bool all() const;
  bool operator[](size_t pos) const;
  std::map<ParallelType, bool> getMap() const;

 private:
  ParallelTypeBitmap(const std::bitset<num_p_type>& bs) : bitset_(bs) {}
  std::bitset<num_p_type> bitset_;
  const static std::unordered_map<ParallelType, int, TypeHash> pt_to_offset_;
  const static std::unordered_map<int, ParallelType> offset_to_pt_;
};

ParallelTypeBitmap operator&(
    const ParallelTypeBitmap& lhs,
    const ParallelTypeBitmap& rhs);

ParallelTypeBitmap operator|(
    const ParallelTypeBitmap& lhs,
    const ParallelTypeBitmap& rhs);

ParallelTypeBitmap operator^(
    const ParallelTypeBitmap& lhs,
    const ParallelTypeBitmap& rhs);

//! Returns a ParallelTypeBitmap representing which domain needs
//! blockBroadcast.
//!
//! Even when a domain is broadcast and parallelized, it does not need
//! blockBroadcast unless it is predicated.
ParallelTypeBitmap getParallelBroadcastDomains(
    const kir::Val* bop_out,
    const kir::ThreadPredicateMap& preds);

} // namespace ir_utils

namespace loop_utils {

// I wanted to make the tv's in these util functions constant, but that started
// a long const-ness project going into TensorView (making functions const
// there) then into lower_loops where we sort exprs.
// TODO: We should fix this when we have some time.

// Figure out which loop the allocation needs to be in. Returns nullptr if
// outside the first loop in loops. Also find out which index in tv the
// first dimension that needs to be allocated is. Meaning we need to allocate
// that local axis and above.
std::pair<kir::ForLoop*, int64_t> getAllocPoint(
    TensorView* tv,
    const std::vector<kir::ForLoop*>& loops);

// Go through exprs mapping root domains from producer to consumer. Provides a
// ground truth for how root domains map through our expressions. Needed for
// unrolling.
//
// TODO(kir): this is only used by UnrollPass, move it there
//
IterDomainMap p2cRootMap(const std::vector<Expr*>& exprs);

} // namespace loop_utils
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch

#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

namespace torch {
namespace jit {
namespace fuser {

namespace {

// A merge is contiguous if:
//   Inputs of outer are to the left in the root domain of the inputs of RHS.
//   All inputs are contiguous in the root domain:
//     - All marked as contiguous
//     - Only gaps between inputs are broadcast or reductoin dims
//   There are no split transformations performed on outer or inner
//   All transformations on outer or inner are contiguous merges
// If this criteria holds, then we can index the input root domains of this
// merge with the indexing provided to the output of the merge in the backward
// index pass

class ContigIDs : public OptInDispatch {
 private:
  using OptInDispatch::handle;

  // Mark if ids are result of contigous merges
  std::unordered_set<IterDomain*> contig_ids;
  const std::vector<IterDomain*>& root_domain_;
  const std::vector<bool>& root_contiguity_;
  std::unordered_map<IterDomain*, bool> is_contig_root;

  ContigIDs() = delete;

  ContigIDs(
      const std::vector<IterDomain*>& ids,
      const std::vector<IterDomain*>& _root_domain,
      const std::vector<bool>& _root_contiguity)
      : root_domain_(_root_domain), root_contiguity_(_root_contiguity) {
    if (ids.empty()) {
      return;
    }

    TORCH_INTERNAL_ASSERT(
        root_domain_.size() == root_contiguity_.size(),
        "Arguments don't match ",
        root_domain_.size(),
        " != ",
        root_contiguity_.size());

    for (size_t i = 0; i < root_domain_.size(); i++) {
      if (root_contiguity_[i]) {
        contig_ids.emplace(root_domain_[i]);
      }
      is_contig_root[root_domain_[i]] = root_contiguity_[i];
    }

    auto exprs = ExprSort::getExprs(ids[0]->fusion(), {ids.begin(), ids.end()});

    for (auto expr : exprs) {
      handle(expr);
    }
  }

  bool inRoot(const std::vector<IterDomain*>& ids) {
    return std::all_of(ids.begin(), ids.end(), [this](IterDomain* id) {
      return is_contig_root.find(id) != is_contig_root.end();
    });
  }

  bool isContig(IterDomain* id) {
    return contig_ids.find(id) != contig_ids.end();
  }

  // Split outputs are not conitguous, don't need to do anything.
  void handle(Split*) override {}

  void handle(Merge* merge) override {
    // If either input is non-contiguous so is output.
    auto inner = merge->inner();
    auto outer = merge->outer();
    if (!isContig(inner) || !isContig(outer)) {
      return;
    }

    // Grab inputs, make sure they're in root domain, check if they're
    // contiguous.

    auto lhs_inputs =
        ir_utils::iterDomainInputsOfOrderedAs({outer}, root_domain_);
    auto rhs_inputs =
        ir_utils::iterDomainInputsOfOrderedAs({inner}, root_domain_);

    TORCH_INTERNAL_ASSERT(
        inRoot(lhs_inputs) && inRoot(rhs_inputs),
        "Found an invalid merge operation, inputs of its arguments are not in the root domain.");

    std::deque<IterDomain*> ordered_inputs(
        lhs_inputs.begin(), lhs_inputs.end());
    ordered_inputs.insert(
        ordered_inputs.end(), rhs_inputs.begin(), rhs_inputs.end());

    // If any root input is not contig, output is not contig
    if (!(std::all_of(
            ordered_inputs.begin(),
            ordered_inputs.end(),
            [this](IterDomain* id) { return is_contig_root.at(id); }))) {
      return;
    }

    std::deque<IterDomain*> root_copy(root_domain_.begin(), root_domain_.end());

    // Forward to first matching argument
    while (!root_copy.empty() && !ordered_inputs.empty()) {
      if (root_copy.front() != ordered_inputs.front()) {
        root_copy.pop_front();
      } else {
        break;
      }
    }

    // Forward through all matching arguments
    while (!root_copy.empty() && !ordered_inputs.empty()) {
      if (root_copy.front() == ordered_inputs.front()) {
        root_copy.pop_front();
        ordered_inputs.pop_front();
        // We probably should be able to make access contiguous through
        // reduction domains, however, for now it's causing issues in predicate
        // generation. See test: ReductionSchedulerMultiDimNonFastest
        //  } else if (
        //     root_copy.front()->isReduction() ||
        //     root_copy.front()->isBroadcast()) {
        //   root_copy.pop_front();
      } else {
        break;
      }
    }

    // If we matched all inputs, the output is contiguous
    if (ordered_inputs.empty()) {
      contig_ids.emplace(merge->out());
    }
  }

 public:
  // Check through thie history of ids whose inputs map to root_domain with
  // contiguity root_contiguity. Return unordered_set of all merges that are
  // contiguous.
  static std::unordered_set<IterDomain*> find(
      const std::vector<IterDomain*>& ids,
      const std::vector<IterDomain*>& root_domain,
      const std::vector<bool>& root_contiguity) {
    ContigIDs finder(ids, root_domain, root_contiguity);
    return finder.contig_ids;
  }
};

// Take a set of ranges on a domain and backward proipagate them to figure out
// the extent of the root domain axes.
class RangeCompute : public BackwardVisitor {
 private:
  using BackwardVisitor::handle;

  void handle(Split* split) override {
    auto in_id = split->in();
    auto outer_id = split->outer();
    auto inner_id = split->inner();

    auto outer_it = range_map_.find(outer_id);
    auto inner_it = range_map_.find(inner_id);
    if (outer_it == range_map_.end() || inner_it == range_map_.end())
      return;

    auto outer_range = outer_it->second;
    auto inner_range = inner_it->second;

    Val* extent = nullptr;

    bool has_zero = outer_range->isZeroInt() || inner_range->isZeroInt();

    bool both_zero = outer_range->isZeroInt() && inner_range->isZeroInt();

    bool zero_merged_in = has_zero ||
        zero_merged_id.find(outer_id) != zero_merged_id.end() ||
        zero_merged_id.find(inner_id) != zero_merged_id.end();

    if (zero_merged_in) {
      zero_merged_id.emplace(in_id);
    }

    if (both_zero) {
      range_map_[in_id] = new Int(0);
    } else if (has_zero) {
      range_map_[in_id] = outer_range->isZeroInt() ? inner_range : outer_range;
    } else if (zero_merged_in) {
      range_map_[in_id] = mul(outer_range, inner_range);
    } else {
      range_map_[in_id] = in_id->extent();
    }
  }

  void handle(Merge* merge) override {
    auto out_id = merge->out();
    auto outer_id = merge->outer();
    auto inner_id = merge->inner();

    auto out_it = range_map_.find(out_id);
    if (out_it == range_map_.end())
      return;

    auto out_range = out_it->second;

    if (contig_ids.find(out_id) != contig_ids.end()) {
      auto input_ids =
          ir_utils::iterDomainInputsOfOrderedAs({out_id}, td_->getRootDomain());

      // Shouldn't hit this, but don't want to segfault if somehow we do.
      TORCH_INTERNAL_ASSERT(!input_ids.empty());

      for (auto root_id : input_ids) {
        range_map_[root_id] = new Int(0);
      }

      range_map_[*(input_ids.end() - 1)] = out_range;
      return;
    }

    // If there was a 0 merged in here due to a split just move the extent to
    // the right
    if (zero_merged_id.find(out_id) != zero_merged_id.end()) {
      range_map_[outer_id] = new Int(0);
      range_map_[inner_id] = out_range;
    } else {
      range_map_[outer_id] = merge->outer()->extent();
      range_map_[inner_id] = merge->inner()->extent();
    }
  }

  void handle(Expr* e) override {
    switch (e->getExprType().value()) {
      case (ExprType::Split):
      case (ExprType::Merge):
        break;
      default:
        TORCH_INTERNAL_ASSERT(
            false, "Invalid expr type found in transform traversal.");
    }
    BackwardVisitor::handle(e);
  }

  RangeCompute(
      const TensorDomain* _td,
      const std::vector<Val*>& ranges,
      std::vector<bool> _root_contiguity)
      : td_(_td) {
    contig_ids =
        ContigIDs::find(td_->domain(), td_->getRootDomain(), _root_contiguity);

    if (td_->nDims() == 0 || ranges.empty()) {
      ranges_.push_back(new Int(0));
      return;
    }

    // TODO: We will always provide reduction ranges, even though they may be 0

    // We may or may not have ranges associated with reductions.
    const bool exclude_reduction = td_->nDims() > ranges.size();

    TORCH_INTERNAL_ASSERT(
        td_->noReductions().size() == ranges.size() ||
            td_->nDims() == ranges.size(),
        "For IndexCompute the number of axes should match the number of dimensions in the TensorDomain.");

    {
      size_t i = 0;
      for (auto id : td_->domain()) {
        if (exclude_reduction && id->isReduction())
          continue;
        range_map_[id] = ranges[i++];
      }
    }

    const std::vector<Val*> domain_vals(
        td_->domain().begin(), td_->domain().end());

    // Run the split/merge operations backwards. This will modify the range_map_
    // so it can be used to index the root TensorDomain. Each entry in the root
    // TensorDomain should have an entry in range_map_ We might not want to run
    // these ranges at the root of the domain, but actually at the rfactor
    // root. Fortunately we can run them all the way back, but grab the ranges
    // from the map at the rfactor IterDomains.
    traverseFrom(ranges[0]->fusion(), domain_vals, false);

    // TODO: Don't exclude reduction axes
    auto root_dom = td_->getMaybeRFactorDomain();
    for (auto id : root_dom) {
      if (exclude_reduction && id->isReduction()) {
        continue;
      } else if (id->getIterType() == IterType::BroadcastWithStride) {
        // TODO: Why not do this for any broadcast dim? Would they be non-zero?
        ranges_.push_back(new Int(1));
      } else {
        auto it = range_map_.find(id);
        TORCH_INTERNAL_ASSERT(
            it != range_map_.end(),
            "Error during index compute, missed computing a value.");
        ranges_.push_back(it->second);
      }
    }
  }

  // Tensor domain we're mapping back to root
  const TensorDomain* td_;
  // Map we update as we propagate backward
  std::unordered_map<IterDomain*, Val*> range_map_;
  // Starting with input ranges, returning as root ranges
  std::vector<Val*> ranges_;
  // IDs that are result of contiguous merges
  std::unordered_set<IterDomain*> contig_ids;
  // IDs that have a 0 merged back into them, we can't map these dims back to
  // the original id->extent.
  std::unordered_set<IterDomain*> zero_merged_id;

 public:
  static std::vector<Val*> get(
      const TensorDomain* _td,
      const std::vector<Val*>& _ranges,
      const std::vector<bool>& _root_contiguity) {
    RangeCompute rc(_td, _ranges, _root_contiguity);
    return rc.ranges_;
  }
};

} // namespace

void IndexCompute::handle(Split* split) {
  auto in_id = split->in();
  auto outer_id = split->outer();
  auto inner_id = split->inner();

  auto outer_it = index_map_.find(outer_id);
  auto inner_it = index_map_.find(inner_id);
  if (outer_it == index_map_.end() || inner_it == index_map_.end())
    return;

  auto outer_ind = outer_it->second;
  auto inner_ind = inner_it->second;

  bool outer_zero = outer_ind->isZeroInt();
  bool inner_zero = inner_ind->isZeroInt();

  bool outer_bcast = outer_id->isBroadcast();
  bool inner_bcast = inner_id->isBroadcast();

  // Zero inds because a dim is bcast is part of normal traversal, if it's not
  // bcast but is zero ind then it's from local or smem. In the latter case we
  // want to propagate this property.
  if ((outer_zero && !outer_bcast) || (inner_zero && !inner_bcast) ||
      hasZeroMerged(inner_id) || hasZeroMerged(outer_id)) {
    zero_merged_in_.emplace(in_id);
  } else {
    // Maybe clear in_id as it could have been mapped over from another
    // IndexCompute. Uncertain if this is needed but seems to be safe.
    if (hasZeroMerged(in_id)) {
      zero_merged_in_.erase(in_id);
    }
  }

  if (outer_zero && inner_zero) {
    index_map_[in_id] = new Int(0);
  } else if (outer_zero) {
    index_map_[in_id] = inner_ind;
    zero_merged_in_.emplace(in_id);
    extent_map_[in_id] = getExtent(inner_id);
  } else if (inner_zero) {
    index_map_[in_id] = outer_ind;
    zero_merged_in_.emplace(in_id);
    extent_map_[in_id] = getExtent(outer_id);
  } else {
    index_map_[in_id] = add(mul(outer_ind, getExtent(inner_id)), inner_ind);
  }
}

void IndexCompute::handle(Merge* merge) {
  auto out_id = merge->out();
  auto outer_id = merge->outer();
  auto inner_id = merge->inner();

  auto out_it = index_map_.find(out_id);
  if (out_it == index_map_.end())
    return;

  auto out_ind = out_it->second;

  auto zero = new Int(0);

  if (out_ind->isZeroInt()) {
    index_map_[outer_id] = zero;
    index_map_[inner_id] = zero;
    extent_map_[outer_id] = zero;
    extent_map_[inner_id] = zero;
    return;
  }

  if (!hasZeroMerged(out_id) && contig_ids.find(out_id) != contig_ids.end()) {
    auto input_ids =
        ir_utils::iterDomainInputsOfOrderedAs({out_id}, td_->getRootDomain());

    // Shouldn't hit this, but don't want to segfault if somehow we do.
    TORCH_INTERNAL_ASSERT(!input_ids.empty());

    for (auto root_id : input_ids) {
      index_map_[root_id] = zero;
    }

    index_map_[*(input_ids.end() - 1)] = out_ind;
    return;
  }

  Val* inner_extent = getExtent(inner_id);
  Val* outer_extent = getExtent(outer_id);

  if (inner_id->isBroadcast() && inner_extent->isOneInt()) {
    index_map_[outer_id] = out_ind;
    index_map_[inner_id] = zero;

    extent_map_[outer_id] = getExtent(out_id);

  } else if (outer_id->isBroadcast() && outer_extent->isOneInt()) {
    index_map_[outer_id] = zero;

    index_map_[inner_id] = out_ind;
    extent_map_[inner_id] = getExtent(out_id);

  } else if (hasZeroMerged(out_id)) {
    index_map_[inner_id] = out_ind;
    extent_map_[inner_id] = getExtent(out_id);

    index_map_[outer_id] = zero;
    extent_map_[outer_id] = zero;

    zero_merged_in_.emplace(inner_id);
    zero_merged_in_.emplace(outer_id);

  } else {
    Val* I = inner_extent;

    Val* outer_ind = div(out_ind, I);
    Val* inner_ind = mod(out_ind, I);

    index_map_[outer_id] = outer_ind;
    index_map_[inner_id] = inner_ind;
  }
}

void IndexCompute::handle(Expr* e) {
  switch (e->getExprType().value()) {
    case (ExprType::Split):
    case (ExprType::Merge):
      break;
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Invalid expr type found in transform traversal.");
  }
  BackwardVisitor::handle(e);
}

// Otherwise warning on runBackward as it hides an overloaded virtual
// using TransformIter::runBackward;
IndexCompute::IndexCompute(
    const TensorDomain* _td,
    const std::unordered_map<IterDomain*, Val*>& initial_index_map,
    const std::unordered_map<IterDomain*, Val*>& _extent_map,
    const std::unordered_set<IterDomain*>& _zero_merged_in)
    : td_(_td),
      index_map_(initial_index_map),
      extent_map_(_extent_map),
      zero_merged_in_(_zero_merged_in) {
  const std::vector<Val*> domain_vals(
      td_->domain().begin(), td_->domain().end());

  traverseFrom(td_->fusion(), domain_vals, false);
}

Val* IndexCompute::getExtent(IterDomain* id) {
  if (extent_map_.find(id) != extent_map_.end()) {
    return extent_map_.at(id);
  } else {
    return id->extent();
  }
}

bool IndexCompute::hasZeroMerged(IterDomain* id) {
  return zero_merged_in_.find(id) != zero_merged_in_.end();
}

IndexCompute::IndexCompute(
    const TensorDomain* _td,
    const std::vector<Val*>& indices,
    std::vector<bool> root_contiguity,
    bool ignore_rfactor)
    : td_(_td), extent_map_(std::unordered_map<IterDomain*, Val*>()) {
  contig_ids =
      ContigIDs::find(td_->domain(), td_->getRootDomain(), root_contiguity);

  if (td_->nDims() == 0 || indices.empty()) {
    indices_.push_back(new Int(0));
    return;
  }

  // TODO: We will always provide reduction indices, even though they may be 0

  // We may or may not have indices associated with reductions.
  const bool exclude_reduction = td_->nDims() > indices.size();

  TORCH_INTERNAL_ASSERT(
      td_->noReductions().size() == indices.size() ||
          td_->nDims() == indices.size(),
      "For IndexCompute the number of axes should match the number of dimensions in the TensorDomain.");

  {
    size_t i = 0;
    for (auto id : td_->domain()) {
      if (exclude_reduction && id->isReduction())
        continue;
      index_map_[id] = indices[i++];
    }
  }

  const std::vector<Val*> domain_vals(
      td_->domain().begin(), td_->domain().end());

  // Run the split/merge operations backwards. This will modify the index_map_
  // so it can be used to index the root TensorDomain. Each entry in the root
  // TensorDomain should have an entry in index_map_ We might not want to run
  // these indices at the root of the domain, but actually at the rfactor root.
  // Fortunately we can run them all the way back, but grab the indices from the
  // map at the rfactor IterDomains.
  traverseFrom(td_->fusion(), domain_vals, false);

  // TODO: Don't exclude reduction axes
  auto root_dom =
      ignore_rfactor ? td_->getRootDomain() : td_->getMaybeRFactorDomain();
  for (auto id : root_dom) {
    if (exclude_reduction && id->isReduction()) {
      continue;
    } else if (id->getIterType() == IterType::BroadcastWithStride) {
      // TODO: Why not do this for any broadcast dim? Would they be non-zero?
      indices_.push_back(new Int(0));
    } else {
      auto it = index_map_.find(id);
      TORCH_INTERNAL_ASSERT(
          it != index_map_.end(),
          "Error during index compute, missed computing a value.");
      indices_.push_back(it->second);
    }
  }
}

IndexCompute IndexCompute::updateIndexCompute(
    const TensorDomain* new_td,
    std::unordered_map<IterDomain*, IterDomain*> id_map,
    std::unordered_map<IterDomain*, Val*> new_index_entries) {
  std::unordered_map<IterDomain*, Val*> updated_index_map(new_index_entries);
  std::unordered_map<IterDomain*, Val*> updated_extent_map;
  std::unordered_set<IterDomain*> updated_zero_merged_in;

  for (auto id_entry : id_map) {
    IterDomain* prev_id = id_entry.first;
    IterDomain* new_id = id_entry.second;

    if (index_map_.find(prev_id) != index_map_.end()) {
      updated_index_map[new_id] = index_map_.at(prev_id);
    }

    updated_extent_map[new_id] = getExtent(prev_id);

    if (zero_merged_in_.find(prev_id) != zero_merged_in_.end()) {
      updated_zero_merged_in.emplace(new_id);
    }
  }

  return IndexCompute(
      new_td, updated_index_map, updated_extent_map, updated_zero_merged_in);
}

std::vector<Val*> IndexCompute::get(
    const TensorDomain* td,
    const std::vector<Val*>& _indices,
    const std::vector<bool>& _root_contiguity,
    bool ignore_rfactor) {
  IndexCompute ic(td, _indices, _root_contiguity, ignore_rfactor);
  return ic.indices_;
}

std::vector<bool> IndexCompute::contiguityAnd(
    const std::vector<bool>& contig1,
    const std::vector<bool>& contig2) {
  TORCH_INTERNAL_ASSERT(
      contig1.size() == contig2.size(),
      "Called contiguityAnd with mismatched vectors.");

  std::vector<bool> contig_result;
  std::transform(
      contig1.begin(),
      contig1.end(),
      contig2.begin(),
      std::back_inserter(contig_result),
      std::logical_and<>());
  return contig_result;
}

// TODO: use new mapping functions
// This mapping might need to go through rfactor, unclear
std::vector<bool> IndexCompute::contiguityPasC(
    TensorDomain* producer,
    TensorDomain* consumer) {
  const std::vector<bool>& producer_contiguity = producer->contiguity();
  std::vector<bool> as_consumer_contiguity;

  auto c_root = consumer->getRootDomain();
  auto p_root = producer->getRootDomain();

  size_t p_ind = 0;
  size_t c_ind = 0;
  while (p_ind < p_root.size()) {
    if (p_root[p_ind]->isReduction()) {
      p_ind++;
    } else if (
        c_root[c_ind]->isBroadcast() &&
        p_root[p_ind]->getIterType() != c_root[c_ind]->getIterType()) {
      c_ind++;
      as_consumer_contiguity.push_back(false);
    } else {
      as_consumer_contiguity.push_back(producer_contiguity[p_ind]);
      c_ind++;
      p_ind++;
    }
  }

  while (c_ind < c_root.size()) {
    as_consumer_contiguity.push_back(false);
    c_ind++;
  }

  return as_consumer_contiguity;
}

namespace {

std::deque<TensorView*> getComputeAtTVStackFrom(TensorView* from_tv) {
  // What's the computeAt root tensor view in this operation
  // This tensor is the terminating tensor in the computeAT dag from consumer
  auto end_tv = from_tv->getComputeAtAxis(0).second;

  // grab all tensor views from producer_tv -> computeAtRoot
  std::deque<TensorView*> tv_stack;

  // Then immediate consumer
  auto running_tv = from_tv;

  // Follow computeAt path until we hit end_tv
  while (running_tv != end_tv) {
    TORCH_INTERNAL_ASSERT(running_tv->hasComputeAt());
    tv_stack.push_front(running_tv);
    running_tv = running_tv->getComputeAtView();
  }

  tv_stack.push_front(end_tv);

  return tv_stack;
}

template <typename T1, typename T2>
void print_map(std::unordered_map<T1, T2> map) {
  std::cout << "{ " << std::endl;
  for (auto entry : map) {
    std::cout << "  ( " << entry.first << " -> " << entry.second << " ) "
              << std::endl;
  }
  std::cout << " }" << std::endl;
}

template <typename T1, typename T2>
void print_map_inline(std::unordered_map<T1, T2> map) {
  IRPrinter print(std::cout);
  std::cout << "{ " << std::endl;
  for (auto entry : map) {
    std::cout << "  ( " << entry.first << " -> ";
    print.print_inline(entry.second);
    std::cout << " ) " << std::endl;
  }
  std::cout << " }" << std::endl;
}

template <typename T1>
void print_set(std::unordered_set<T1> set) {
  std::cout << "{ " << std::endl;
  for (auto entry : set) {
    std::cout << "  ( " << entry << " ) " << std::endl;
  }
  std::cout << " }" << std::endl;
}

std::pair<
    std::unordered_map<IterDomain*, Val*>,
    std::unordered_map<IterDomain*, Val*>>
generateIndexAndExtentMap(
    std::deque<TensorView*> tv_stack,
    std::deque<kir::ForLoop*> loops,
    const std::unordered_map<kir::ForLoop*, Val*>& loop_to_ind_map) {
  // Go through our stack, and map the intermediate IterDomains from common
  // transformations from consumer to producer
  std::deque<std::unordered_map<IterDomain*, IterDomain*>> ID_maps_c2p;

  for (size_t i = 0; i + 1 < tv_stack.size(); i++) {
    auto c_tv = tv_stack[i];
    auto p_tv = tv_stack[i + 1];

    // Map root ID's from consumer to producer
    auto c2p_root_map =
        TensorDomain::mapRootCtoP(c_tv->domain(), p_tv->domain());

    // Look for matching ID transformations in producer and consumer...
    BestEffortReplay replay(
        p_tv->domain()->domain(), c_tv->domain()->domain(), c2p_root_map);

    // and grab the intermediate IterDomain map.
    ID_maps_c2p.push_back(replay.getReplay());
  }

  if (tv_stack.empty())
    return std::make_pair(
        std::unordered_map<IterDomain*, Val*>(),
        std::unordered_map<IterDomain*, Val*>());

  // Setup initial IndexCompute:
  auto tv = tv_stack.front();
  tv_stack.pop_front();
  auto td = tv->domain()->domain();

  // Map from all IterDomain's to corresponding index as we process each tv in
  // the stack
  std::unordered_map<IterDomain*, Val*> initial_index_map;

  // Match loops to this TV if the loop matchis this TV's ID (could reduce
  // complexity here)
  while (!loops.empty() &&
         std::find(td.begin(), td.end(), loops.front()->iter_domain()) !=
             td.end()) {
    TORCH_INTERNAL_ASSERT(
        loop_to_ind_map.find(loops.front()) != loop_to_ind_map.end());
    initial_index_map[loops.front()->iter_domain()] =
        loop_to_ind_map.at(loops.front());
    loops.pop_front();
  }

  IndexCompute index_compute(
      tv->domain(),
      initial_index_map,
      std::unordered_map<IterDomain*, Val*>(),
      std::unordered_set<IterDomain*>());

  // Go through the tv entire stack
  while (!tv_stack.empty()) {
    // Grab the TV
    tv = tv_stack.front();
    tv_stack.pop_front();
    td = tv->domain()->domain();

    std::unordered_map<IterDomain*, Val*> new_indices;

    // Match loops to this TV if the loop matchis this TV's ID (could reduce
    // complexity here)
    while (!loops.empty() &&
           std::find(td.begin(), td.end(), loops.front()->iter_domain()) !=
               td.end()) {
      TORCH_INTERNAL_ASSERT(
          loop_to_ind_map.find(loops.front()) != loop_to_ind_map.end());
      new_indices[loops.front()->iter_domain()] =
          loop_to_ind_map.at(loops.front());
      loops.pop_front();
    }

    if (!ID_maps_c2p.empty()) {
      index_compute = index_compute.updateIndexCompute(
          tv->domain(), ID_maps_c2p.front(), new_indices);
      ID_maps_c2p.pop_front();
    }
  }
  return std::make_pair(index_compute.indexMap(), index_compute.extentMap());
}

} // namespace

kir::TensorIndex* Index::getGlobalProducerIndex(
    TensorView* producer_tv,
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, IterDomain*>& p2c_root_map) {
  // Replay producer to look like consumer so we can index on producer since our
  // loop nests look like consumer
  auto producerAsC = TransformReplay::replayPasC(
                         producer_tv->domain(), consumer_tv->domain(), -1)
                         .first;

  // Make the actual producer_tv look like consumer while we do the indexing
  // math in this function
  ir_utils::TVDomainGuard domain_guard(producer_tv, producerAsC);

  // grab all tensor views from producer_tv <- computeAtRoot
  std::deque<TensorView*> tv_stack = getComputeAtTVStackFrom(consumer_tv);
  tv_stack.push_back(producer_tv);

  std::unordered_map<kir::ForLoop*, Val*> loop_to_ind_map;
  std::transform(
      loops.begin(),
      loops.end(),
      std::inserter(loop_to_ind_map, loop_to_ind_map.begin()),
      [](kir::ForLoop* fl) { return std::make_pair(fl, fl->index()); });

  auto index_map = generateIndexAndExtentMap(
                       tv_stack,
                       std::deque<kir::ForLoop*>(loops.begin(), loops.end()),
                       loop_to_ind_map)
                       .first;

  // Indices should now be mapped onto IterDomains in producer, so just grab
  // and use them.
  auto zero = new Int(0);

  auto root_dom = producer_tv->getMaybeRFactorDomain();

  bool inner_most_dim_contig =
      root_dom[root_dom.size() - 1]->getIterType() == IterType::Iteration &&
      producer_tv->domain()->contiguity()[root_dom.size() - 1];

  // Global striding
  int64_t stride_i = 0;
  std::vector<Val*> strided_inds;
  for (size_t i = 0; i < root_dom.size(); i++) {
    if (root_dom[i]->isReduction() ||
        root_dom[i]->getIterType() == IterType::BroadcastWithoutStride) {
      continue;
    } else if (root_dom[i]->getIterType() == IterType::BroadcastWithStride) {
      stride_i++;
      continue;
    }

    TORCH_INTERNAL_ASSERT(index_map.find(root_dom[i]) != index_map.end());
    auto root_ind = index_map.at(root_dom[i]);

    if (i == root_dom.size() - 1 && inner_most_dim_contig) {
      strided_inds.push_back(root_ind);
    } else if (root_ind->isZeroInt()) {
      stride_i++;
    } else {
      std::stringstream ss;
      ss << "T" << producer_tv->name() << ".stride[" << stride_i++ << "]";
      strided_inds.push_back(
          mul(root_ind, new NamedScalar(ss.str(), DataType::Int)));
    }
  }

  if (strided_inds.size() == 0)
    strided_inds.push_back(new Int(0));

  return new kir::TensorIndex(producer_tv, strided_inds);
}

namespace {

std::unordered_map<kir::ForLoop*, Val*> indexMapFromTV(
    TensorView* tv,
    const std::vector<kir::ForLoop*>& loops) {
  auto alloc_point = loop_utils::getAllocPoint(tv, loops);
  auto alloc_loop = alloc_point.first;

  bool within_alloc = false;
  if (alloc_loop == nullptr) {
    within_alloc = true;
  }

  Val* zero = new Int(0);

  bool is_shared = tv->getMemoryType() == MemoryType::Shared;
  bool is_local = tv->getMemoryType() == MemoryType::Local;

  std::unordered_map<kir::ForLoop*, Val*> loop_to_ind_map;

  for (auto loop : loops) {
    if (!within_alloc) {
      loop_to_ind_map[loop] = zero;
    } else if (loop->iter_domain()->isBlockDim() && is_shared) {
      loop_to_ind_map[loop] = zero;
    } else if (loop->iter_domain()->isThread() && is_local) {
      loop_to_ind_map[loop] = zero;
    } else {
      loop_to_ind_map[loop] = loop->index();
    }

    if (!within_alloc && loop == alloc_loop) {
      within_alloc = true;
    }
  }
  return loop_to_ind_map;
}
} // namespace

// Producer index for either shared or local memory
kir::TensorIndex* Index::getProducerIndex_impl(
    TensorView* producer_tv,
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, IterDomain*>& p2c_root_map) {
  // producer_tv->domain() is not replayed as the loop strucutre we were
  // provided, so replay it to match consumer_tv which is.
  auto producerAsC = TransformReplay::replayPasC(
                         producer_tv->domain(), consumer_tv->domain(), -1)
                         .first;

  // Set producer_tv with the domain replayed as consumer to grab the right
  // indices. The guard will reset the domain when this scope ends.
  ir_utils::TVDomainGuard domain_guard(producer_tv, producerAsC);

  // grab all tensor views from producer_tv <- computeAtRoot
  std::deque<TensorView*> tv_stack = getComputeAtTVStackFrom(consumer_tv);
  tv_stack.push_back(producer_tv);

  std::unordered_map<kir::ForLoop*, Val*> loop_to_ind_map =
      indexMapFromTV(producer_tv, loops);

  auto index_and_extent_map = generateIndexAndExtentMap(
      tv_stack,
      std::deque<kir::ForLoop*>(loops.begin(), loops.end()),
      loop_to_ind_map);
  auto index_map = index_and_extent_map.first;
  auto extent_map = index_and_extent_map.second;

  // Indices should now be mapped onto IterDomains in producer, so just grab
  // and use them.
  auto zero = new Int(0);

  auto root_dom = producer_tv->getMaybeRFactorDomain();

  std::vector<Val*> strided_inds;

  for (size_t i = 0; i < root_dom.size(); i++) {
    if (root_dom[i]->isReduction() || root_dom[i]->isBroadcast()) {
      continue;
    }

    TORCH_INTERNAL_ASSERT(index_map.find(root_dom[i]) != index_map.end());
    auto root_ind_i = index_map.at(root_dom[i]);

    if (root_ind_i->isZeroInt()) {
      continue;
    }

    // Compute striding for this index.
    Val* stride = nullptr;
    for (size_t j = i + 1; j < root_dom.size(); j++) {
      if (root_dom[j]->isBroadcast() || root_dom[j]->isReduction()) {
        continue;
      }

      TORCH_INTERNAL_ASSERT(
          index_map.find(root_dom[j]) != index_map.end() &&
          extent_map.find(root_dom[j]) != extent_map.end());
      auto root_ind_j = index_map.at(root_dom[j]);
      auto root_ext_j = extent_map.at(root_dom[j]);

      if (!root_ind_j->isZeroInt()) {
        if (stride == nullptr) {
          stride = root_ext_j;
        } else {
          stride = mul(stride, root_ext_j);
        }
      }
    }

    if (stride != nullptr) {
      strided_inds.push_back(mul(root_ind_i, stride));
    } else {
      strided_inds.push_back(root_ind_i);
    }
  }

  if (strided_inds.size() == 0)
    strided_inds.push_back(new Int(0));

  return new kir::TensorIndex(producer_tv, strided_inds);
}

kir::TensorIndex* Index::getGlobalConsumerIndex(
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, IterDomain*>& p2c_root_map) {
  // grab all tensor views from producer_tv <- computeAtRoot
  std::deque<TensorView*> tv_stack = getComputeAtTVStackFrom(consumer_tv);

  std::unordered_map<kir::ForLoop*, Val*> loop_to_ind_map;
  std::transform(
      loops.begin(),
      loops.end(),
      std::inserter(loop_to_ind_map, loop_to_ind_map.begin()),
      [](kir::ForLoop* fl) { return std::make_pair(fl, fl->index()); });

  auto index_map = generateIndexAndExtentMap(
                       tv_stack,
                       std::deque<kir::ForLoop*>(loops.begin(), loops.end()),
                       loop_to_ind_map)
                       .first;

  // Indices should now be mapped onto IterDomains in consumer, so just grab
  // and use them.
  auto zero = new Int(0);

  auto root_dom = consumer_tv->getMaybeRFactorDomain();

  bool inner_most_dim_contig =
      root_dom[root_dom.size() - 1]->getIterType() == IterType::Iteration &&
      consumer_tv->domain()->contiguity()[root_dom.size() - 1];

  int64_t stride_i = 0;
  std::vector<Val*> strided_inds;
  for (size_t i = 0; i < root_dom.size(); i++) {
    if (root_dom[i]->isReduction() ||
        root_dom[i]->getIterType() == IterType::BroadcastWithoutStride) {
      continue;
    } else if (root_dom[i]->getIterType() == IterType::BroadcastWithStride) {
      stride_i++;
      continue;
    }

    TORCH_INTERNAL_ASSERT(index_map.find(root_dom[i]) != index_map.end());
    auto ind = index_map.at(root_dom[i]);

    if (i == root_dom.size() - 1 && inner_most_dim_contig) {
      strided_inds.push_back(ind);
    } else if (ind->isZeroInt()) {
      stride_i++;
    } else {
      std::stringstream ss;
      ss << "T" << consumer_tv->name() << ".stride[" << stride_i++ << "]";
      strided_inds.push_back(
          mul(ind, new NamedScalar(ss.str(), DataType::Int)));
    }
  }

  if (strided_inds.size() == 0)
    strided_inds.push_back(new Int(0));

  return new kir::TensorIndex(consumer_tv, strided_inds);
}

// Consumer index for either shared or local memory
kir::TensorIndex* Index::getConsumerIndex_impl(
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, IterDomain*>& p2c_root_map) {
  // grab all tensor views from producer_tv <- computeAtRoot
  std::deque<TensorView*> tv_stack = getComputeAtTVStackFrom(consumer_tv);

  std::unordered_map<kir::ForLoop*, Val*> loop_to_ind_map =
      indexMapFromTV(consumer_tv, loops);

  auto index_and_extent_map = generateIndexAndExtentMap(
      tv_stack,
      std::deque<kir::ForLoop*>(loops.begin(), loops.end()),
      loop_to_ind_map);

  auto index_map = index_and_extent_map.first;
  auto extent_map = index_and_extent_map.second;

  // Indices should now be mapped onto IterDomains in producer, so just grab
  // and use them.
  auto zero = new Int(0);

  auto root_dom = consumer_tv->getMaybeRFactorDomain();

  std::vector<Val*> strided_inds;
  for (size_t i = 0; i < root_dom.size(); i++) {
    if (root_dom[i]->isReduction() || root_dom[i]->isBroadcast()) {
      continue;
    }

    TORCH_INTERNAL_ASSERT(index_map.find(root_dom[i]) != index_map.end());
    auto root_ind_i = index_map.at(root_dom[i]);

    if (root_ind_i->isZeroInt()) {
      continue;
    }

    // Compute striding for this index.
    Val* stride = nullptr;
    for (size_t j = i + 1; j < root_dom.size(); j++) {
      if (root_dom[j]->isBroadcast() || root_dom[j]->isReduction()) {
        continue;
      }

      TORCH_INTERNAL_ASSERT(
          index_map.find(root_dom[j]) != index_map.end() &&
          extent_map.find(root_dom[j]) != extent_map.end());
      auto root_ind_j = index_map.at(root_dom[j]);
      auto root_ext_j = extent_map.at(root_dom[j]);

      if (!root_ind_j->isZeroInt()) {
        if (stride == nullptr) {
          stride = root_ext_j;
        } else {
          stride = mul(stride, root_ext_j);
        }
      }
    }

    if (stride != nullptr) {
      strided_inds.push_back(mul(root_ind_i, stride));
    } else {
      strided_inds.push_back(root_ind_i);
    }
  }

  if (strided_inds.size() == 0)
    strided_inds.push_back(new Int(0));

  return new kir::TensorIndex(consumer_tv, strided_inds);
}

// Producer is the inputs of an expression
kir::TensorIndex* Index::getProducerIndex(
    TensorView* producer,
    TensorView* consumer,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, IterDomain*>& p2c_root_map) {
  if (producer->domain()->noReductions().size() == 0) {
    return new kir::TensorIndex(producer, {});
  }

  if (producer->getMemoryType() == MemoryType::Global)
    return getGlobalProducerIndex(producer, consumer, loops, p2c_root_map);
  return getProducerIndex_impl(producer, consumer, loops, p2c_root_map);
}

// Consumer is the output of an expression
kir::TensorIndex* Index::getConsumerIndex(
    TensorView* consumer,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, IterDomain*>& p2c_root_map) {
  if (consumer->domain()->noReductions().size() == 0) {
    return new kir::TensorIndex(consumer, {});
  }

  if (consumer->getMemoryType() == MemoryType::Global)
    return getGlobalConsumerIndex(consumer, loops, p2c_root_map);
  return getConsumerIndex_impl(consumer, loops, p2c_root_map);
}

} // namespace fuser
} // namespace jit
} // namespace torch

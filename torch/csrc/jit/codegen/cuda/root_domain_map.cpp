#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>

#include <sstream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

std::unordered_map<IterDomain*, IterDomain*> RootDomainMap::
    mapProducerToConsumer(
        const TensorDomain* producer,
        const TensorDomain* consumer,
        const std::unordered_set<IterDomain*>& root_dims_to_map) const {
  return map(producer, consumer, root_dims_to_map, true);
}

std::unordered_map<IterDomain*, IterDomain*> RootDomainMap::
    mapConsumerToProducer(
        const TensorDomain* consumer,
        const TensorDomain* producer,
        const std::unordered_set<IterDomain*>& root_dims_to_map) const {
  return map(producer, consumer, root_dims_to_map, false);
}

PairwiseRootDomainMap::PairwiseRootDomainMap(
    const TensorView* producer,
    const TensorView* consumer)
    : producer_tv_(producer), consumer_tv_(consumer) {
  TORCH_INTERNAL_ASSERT(producer != nullptr);
  TORCH_INTERNAL_ASSERT(consumer != nullptr);
  TORCH_INTERNAL_ASSERT(producer->fusion() == consumer->fusion());
  // Make sure they are really a producer and its consumer
  Expr* origin = consumer->getOrigin();
  TORCH_INTERNAL_ASSERT(origin != nullptr);
  TORCH_INTERNAL_ASSERT(
      std::any_of(origin->inputs().begin(), origin->inputs().end(),
                  [producer](const Val* input) {
                    return input == producer;
                  }),
      "Not a producer-consumer pair: ",
      producer,
      ", ",
      consumer);
  if (BroadcastOp* bop = dynamic_cast<BroadcastOp*>(origin)) {
    broadcast_flags_ = bop->getBroadcastDimFlags();
  } else {
    broadcast_flags_ =
        std::vector<bool>(consumer->getRootDomain().size(), false);
  }
}

std::unordered_map<IterDomain*, IterDomain*> PairwiseRootDomainMap::map(
    const TensorDomain* producer,
    const TensorDomain* consumer,
    const std::unordered_set<IterDomain*>& root_dims_to_map,
    bool producer_to_consumer) const {
  // Sanity check that the given producer and consumer domains are
  // really the TensorDomains of the producer and consumer TensorViews
  // given to the constructor.
  TORCH_INTERNAL_ASSERT(
      producer_tv_ == nullptr || producer_tv_->domain() == producer);
  TORCH_INTERNAL_ASSERT(
      consumer_tv_ == nullptr || consumer_tv_->domain() == consumer);

  std::unordered_map<IterDomain*, IterDomain*> dom_map;
  const auto& producer_root = producer->getMaybeRFactorDomain();
  const auto& consumer_root = consumer->getRootDomain();
  size_t itc = 0, itp = 0;
  while (itc < consumer_root.size() && itp < producer_root.size()) {
    IterDomain* producer_id = producer_root[itp];
    IterDomain* consumer_id = consumer_root[itc];

    // When the producer ID is a reduction domain, there should never
    // be any matching domain in the consumer.
    if (producer_id->isReduction()) {
      itp++;
      continue;
    }

    // When the consumer ID is a new broadcast domain, there is no
    // mapping for it.
    if (broadcast_flags_.at(itc)) {
      TORCH_INTERNAL_ASSERT(consumer_id->isBroadcast());
      itc++;
      continue;
    }

    IterDomain* map_key_id = producer_id;
    IterDomain* map_value_id = consumer_id;
    if (!producer_to_consumer) {
      std::swap(map_key_id, map_value_id);
    }

    if (root_dims_to_map.find(map_key_id) != root_dims_to_map.end()) {
      dom_map.insert(std::make_pair(map_key_id, map_value_id));
    }
    itc++;
    itp++;
  }
  return dom_map;
}

std::ostream& PairwiseRootDomainMap::print(std::ostream& os) const {
  return os << "{producer: " << producer_tv_ << ", consumer: " << consumer_tv_
            << ", broadcast_flags: " << broadcast_flags_ << "}";
}

namespace {

template <typename T>
auto ensureMapping(
    T& m,
    const typename T::key_type& key,
    const typename T::mapped_type& init_value) {
  auto it = m.find(key);
  if (it == m.end()) {
    it = m.insert({key, init_value}).first;
  }
  return it;
}

} // namespace

std::ostream& DomainKey::print(std::ostream& os) const {
  std::stringstream ss;
  ss << "{";
  if (td_) {
    ss << td_ << " (root: " << td_->getRootDomain()
       << ", maybe rfactor: " << td_->getMaybeRFactorDomain() << ")";
  } else {
    ss << "null";
  }
  ss << ", ";
  if (id_) {
    ss << id_;
  } else {
    ss << "null";
  }
  if (concrete_id_) {
    ss << " (" << concrete_id_ << ")";
  }
  ss << "}";
  return os << ss.str();
}

UnmappableReductionDomains::UnmappableReductionDomains() {
  Fusion* fusion = FusionGuard::getCurFusion();
  traverse(fusion);
}

void UnmappableReductionDomains::handle(ReductionOp* op) {
  // Builds a map from reduction domains to consumer domains.
  TensorView* out_tv = op->out()->as<TensorView>();
  std::vector<DomainKey> reduction_keys;
  for (const auto id : out_tv->getMaybeRFactorDomain()) {
    if (id->isReduction()) {
      DomainKey key(out_tv->domain(), id);
      reduction_keys.push_back(key);
      reduction_domains_.insert({key, {}});
    }
  }
  auto use_chains = DependencyCheck::getAllUseChains(out_tv);
  for (const auto& chain : use_chains) {
    for (const auto& tv : ir_utils::filterByType<TensorView>(chain)) {
      const auto& root_domain = tv->getRootDomain();
      for (const auto& id : root_domain) {
        DomainKey consumer_key(tv->domain(), id);
        for (const auto& reduction_key : reduction_keys) {
          reduction_domains_.at(reduction_key).insert(consumer_key);
        }
      }
    }
  }
}

bool UnmappableReductionDomains::isReductionOutputMapped(
    const std::vector<DomainKey>& consumer_domains,
    const ComputeAtRootDomainMap& root_map) const {
  for (const auto& kv : reduction_domains_) {
    const DomainKey& reducion_domain = kv.first;
    const DomainKeySet& incompatible_domains = kv.second;
    DomainKey consumer_domain_with_reduction;
    bool reduction_found = false;
    for (const DomainKey& consumer_domain : consumer_domains) {
      if (root_map.canMap(
              consumer_domain.td(),
              consumer_domain.id(),
              reducion_domain.td(),
              reducion_domain.id())) {
        consumer_domain_with_reduction = consumer_domain;
        reduction_found = true;
        break;
      }
    }
    if (!reduction_found) {
      continue;
    }
    // Make sure no incompatible domains will be merged with the reduction
    // domain.
    for (const auto& consumer_domain : consumer_domains) {
      if (consumer_domain == consumer_domain_with_reduction) {
        continue;
      }
      if (std::any_of(
              incompatible_domains.begin(),
              incompatible_domains.end(),
              [&](const DomainKey& incompatible_domain) {
                return root_map.canMap(
                    consumer_domain.td(),
                    consumer_domain.id(),
                    incompatible_domain.td(),
                    incompatible_domain.id());
              })) {
        return true;
      }
    }
  }
  return false;
}

void ComputeAtRootDomainMap::build() {
  // Make sure we start from scratch. Throw away previous results.
  eq_set_.clear();
  bcast_map_.clear();
  new_broadcast_domains_.clear();
  ComputeAtRootDomainMapBuilder builder(*this);
}

bool ComputeAtRootDomainMap::canMap(
    const TensorDomain* td_a,
    const IterDomain* id_a,
    const TensorDomain* td_b,
    const IterDomain* id_b) const {
  TORCH_INTERNAL_ASSERT(
      id_a->getOrigin() == nullptr || id_a->isRFactorProduct(),
      "Non-root domain is not supproted: ",
      id_a);
  TORCH_INTERNAL_ASSERT(
      id_b->getOrigin() == nullptr || id_b->isRFactorProduct(),
      "Non-root domain is not supproted: ",
      id_b);

  if (hasConcretizedDomains(td_a, id_a)) {
    for (const auto& key_a : getConcretizedKeys(td_a, id_a)) {
      if (canMap(key_a, td_b, id_b)) {
        return true;
      }
    }
    return false;
  } else {
    return canMap(DomainKey(td_a, id_a), td_b, id_b);
  }
}

bool ComputeAtRootDomainMap::canMap(
    const DomainKey& key_a,
    const TensorDomain* td_b,
    const IterDomain* id_b) const {
  TORCH_INTERNAL_ASSERT(
      id_b->getOrigin() == nullptr || id_b->isRFactorProduct(),
      "Non-root domain is not supproted: ",
      id_b);

  if (hasConcretizedDomains(td_b, id_b)) {
    for (const auto& key_b_bc : getConcretizedKeys(td_b, id_b)) {
      if (canMap(key_a, key_b_bc)) {
        return true;
      }
    }
    return false;
  } else {
    return canMap(key_a, DomainKey(td_b, id_b));
  }
}

bool ComputeAtRootDomainMap::canMap(
    const DomainKey& key_a,
    const DomainKey& key_b) const {
  return key_a == key_b || eq_set_.areEquivalent(key_a, key_b);
}

void ComputeAtRootDomainMap::setAlias(
    const TensorDomain* td,
    const TensorDomain* td_alias) {
  auto tmp_bcast_map = bcast_map_;
  for (const auto& kv : bcast_map_) {
    const auto& bcast_map_key = kv.first;
    const auto& bcast_concrete_id_set = kv.second;
    if (bcast_map_key.td() == td) {
      DomainKey alias_key(td_alias, bcast_map_key.id());
      tmp_bcast_map.insert({alias_key, bcast_concrete_id_set});
    }
  }
  bcast_map_ = tmp_bcast_map;

  for (const auto& key : eq_set_.getAllElements()) {
    if (key.td() == td) {
      DomainKey alias_key(td_alias, key.id(), key.concreteId());
      eq_set_.join(key, alias_key);
    }
  }

  auto tmp_new_broadcast_domains = new_broadcast_domains_;
  for (const auto& key : new_broadcast_domains_) {
    if (key.td() == td) {
      DomainKey alias_key(td_alias, key.id());
      tmp_new_broadcast_domains.insert(alias_key);
    }
  }
  new_broadcast_domains_ = tmp_new_broadcast_domains;
}

bool ComputeAtRootDomainMap::hasConcretizedDomains(
    const TensorDomain* td,
    const IterDomain* id) const {
  return id->isBroadcast();
}

std::vector<DomainKey> ComputeAtRootDomainMap::getConcretizedKeys(
    const TensorDomain* td,
    const IterDomain* id) const {
  DomainKey key(td, id);
  auto it = bcast_map_.find(key);
  TORCH_INTERNAL_ASSERT(it != bcast_map_.end(), "Not found: ", key);
  std::vector<DomainKey> domains;
  std::transform(
      it->second.begin(),
      it->second.end(),
      std::back_inserter(domains),
      [&](const IterDomain* concrete_id) {
        return DomainKey(td, id, concrete_id);
      });
  return domains;
}

std::unordered_set<const IterDomain*>& ComputeAtRootDomainMap::
    getConcretizedDomains(const TensorDomain* td, const IterDomain* id) {
  DomainKey key(td, id);
  auto it = bcast_map_.find(key);
  TORCH_INTERNAL_ASSERT(it != bcast_map_.end(), "Not found: ", key);
  return it->second;
}

std::unordered_map<IterDomain*, IterDomain*> ComputeAtRootDomainMap::map(
    const TensorDomain* producer,
    const TensorDomain* consumer,
    const std::unordered_set<IterDomain*>& root_dims_to_map,
    bool producer_to_consumer) const {
  const auto& producer_root = producer->getMaybeRFactorDomain();
  const auto& consumer_root = consumer->getRootDomain();
  const TensorDomain* src_td = producer_to_consumer ? producer : consumer;
  const TensorDomain* dst_td = producer_to_consumer ? consumer : producer;
  const auto& src_ids = producer_to_consumer ? producer_root : consumer_root;
  const auto& dst_ids = producer_to_consumer ? consumer_root : producer_root;
  std::unordered_map<IterDomain*, IterDomain*> id_map;
  for (auto& src_id : src_ids) {
    if (root_dims_to_map.find(src_id) == root_dims_to_map.end()) {
      continue;
    }
    bool mapping_found = false;
    for (const auto& dst_id : dst_ids) {
      if (canMap(src_td, src_id, dst_td, dst_id)) {
        TORCH_INTERNAL_ASSERT(
            id_map.insert({src_id, dst_id}).second,
            "Multiple matching ID detected for ",
            src_id);
        mapping_found = true;
      }
    }
    if (mapping_found) {
      continue;
    }
    // Matching ID not found. It's an error unless: src_id is
    // reduction when producer_to_consumer; or src_id is a new
    // broadcast when !producer_to_consumer.
    if ((producer_to_consumer && src_id->isReduction()) ||
        (!producer_to_consumer &&
         new_broadcast_domains_.find(DomainKey(src_td, src_id)) !=
             new_broadcast_domains_.end())) {
      continue;
    }
    TORCH_INTERNAL_ASSERT(
        false,
        "Mapping IterDomain ",
        src_id,
        " of ",
        src_td,
        " not possible as it would require recomputing the source tensor.",
        " Producer root: ",
        producer_root,
        ". Consumer root: ",
        consumer_root);
  }
  return id_map;
}

std::ostream& ComputeAtRootDomainMap::print(std::ostream& os) const {
  return eq_set_.print(os);
}

ComputeAtRootDomainMapBuilder::ComputeAtRootDomainMapBuilder(
    ComputeAtRootDomainMap& root_map)
    : root_map_(root_map) {
  Fusion* fusion = FusionGuard::getCurFusion();
  TORCH_INTERNAL_ASSERT(fusion != nullptr);
  // Set concrete domains for broadcast domains that never get joined
  // with a concrete domain. Just set its own domain as a concrete
  // domain, which is not concrete but is sufficient for this analysis.
  for (const TensorView* output_tv : ir_utils::filterByType<TensorView>(fusion->outputs())) {
    for (const IterDomain* id : output_tv->getRootDomain()) {
      if (id->isBroadcast()) {
        auto it = ensureMapping(
            root_map.bcast_map_, DomainKey(output_tv->domain(), id), {});
        it->second.insert(id);
      }
    }
  }
  traverseFrom(fusion, fusion->outputs(), false);
  if (!pending_map_.empty()) {
    std::stringstream ss;
    ss << "pending map:\n";
    for (auto& kv : pending_map_) {
      ss << "\t" << kv.first << "\n";
      for (auto& dk : kv.second) {
        ss << "\t\t" << dk << "\n";
      }
    }
    std::cerr << ss.str();
  }
  TORCH_INTERNAL_ASSERT(pending_map_.empty());
}

void ComputeAtRootDomainMapBuilder::addToPendingList(
    const DomainKey& producer,
    const DomainKey& consumer) {
  auto it = ensureMapping(pending_map_, producer, {});
  auto& consumer_set = it->second;
  consumer_set.insert(consumer);
}

void ComputeAtRootDomainMapBuilder::setMapped(
    const DomainKey& producer,
    const DomainKey& consumer) {
  root_map_.eq_set_.join(producer, consumer);
}

void ComputeAtRootDomainMapBuilder::setMaybeMapped(
    const TensorDomain* producer_td,
    const IterDomain* producer_id,
    const TensorDomain* consumer_td,
    const IterDomain* consumer_id) {
  const DomainKey producer_key(producer_td, producer_id);
  const DomainKey consumer_key(consumer_td, consumer_id);

  if (producer_id->isBroadcast()) {
    ensureMapping(root_map_.bcast_map_, producer_key, {});
  }

  if (root_map_.hasConcretizedDomains(consumer_td, consumer_id)) {
    TORCH_INTERNAL_ASSERT(producer_id->isBroadcast());
    // Get bcast_map_ entry for consumer_id
    const auto consumer_bcast_domains =
        root_map_.getConcretizedKeys(consumer_td, consumer_id);
    auto& producer_domains =
        root_map_.getConcretizedDomains(producer_td, producer_id);

    // If consumer id is broadcasted, make sure to propagate its concrete_id(s)
    // to producer
    for (const auto& consumer_bcast_key : consumer_bcast_domains) {
      const auto concrete_id = consumer_bcast_key.concreteId();
      const DomainKey producer_bcast_key(producer_td, producer_id, concrete_id);
      producer_domains.insert(concrete_id);
      addToPendingList(producer_bcast_key, consumer_bcast_key);
    }
  } else {
    TORCH_INTERNAL_ASSERT(
        !consumer_id->isBroadcast(),
        "No concrete domain found for a broadcast domain: ",
        consumer_key);
    auto producer_concrete_key = producer_key;
    if (producer_id->isBroadcast()) {
      const auto concrete_id = consumer_id;
      auto& producer_domains =
          root_map_.getConcretizedDomains(producer_td, producer_id);
      producer_concrete_key = DomainKey(producer_td, producer_id, concrete_id);
      producer_domains.insert(concrete_id);
    }
    addToPendingList(producer_concrete_key, consumer_key);
  }
}

void ComputeAtRootDomainMapBuilder::handle(Expr* e) {
  // Avoid visiting expressions multiple times
  if (visited_.find(e) != visited_.end()) {
    return;
  }
  BackwardVisitor::handle(e);
  visited_.insert(e);
}

void ComputeAtRootDomainMapBuilder::mapPointwiseOrReductionOp(Expr* e) {
  if (e->output(0)->getValType() != ValType::TensorView) {
    return;
  }

  // Broadcast is handled separately, so e should never be BroadcastOp.
  TORCH_INTERNAL_ASSERT(e->getExprType() != ExprType::BroadcastOp);

  TORCH_INTERNAL_ASSERT(e->outputs().size() == 1);
  const TensorView* out_tv = e->output(0)->as<TensorView>();
  const TensorDomain* out_td = out_tv->domain();
  const auto& out_root = out_td->getRootDomain();

  // Record equalities from output to all the inputs
  // ignores un-concretizable broadcasts
  for (auto* i : ir_utils::filterByType<TensorView>(e->inputs())) {
    const TensorDomain* in_td = i->domain();
    std::vector<IterDomain*> in_root =
        TensorDomain::noReductions(i->getMaybeRFactorDomain());
    TORCH_INTERNAL_ASSERT(in_root.size() == out_root.size());
    for (size_t it = 0; it < in_root.size(); it++) {
      setMaybeMapped(in_td, in_root[it], out_td, out_root[it]);
    }
  }
}

void ComputeAtRootDomainMapBuilder::handle(BroadcastOp* op) {
  const TensorDomain* in_td = op->in()->as<TensorView>()->domain();
  const TensorDomain* out_td = op->out()->as<TensorView>()->domain();
  const auto in_root = TensorDomain::noReductions(in_td->getRootDomain());
  const auto& out_root = out_td->getRootDomain();
  const auto& bcast_dim_flags = op->getBroadcastDimFlags();
  TORCH_INTERNAL_ASSERT(
      out_root.size() == bcast_dim_flags.size(),
      "dim flags: ",
      bcast_dim_flags,
      ", out root: ",
      out_root);
  auto in_it = in_root.begin();
  auto out_it = out_root.begin();
  while (in_it != in_root.end() && out_it != out_root.end()) {
    if (bcast_dim_flags.at(std::distance(out_root.begin(), out_it))) {
      // new broadcast dim. No matching dimension in the input
      // tensor.
      root_map_.new_broadcast_domains_.insert(DomainKey(out_td, *out_it));
      ++out_it;
      continue;
    }
    setMaybeMapped(in_td, *in_it, out_td, *out_it);
    ++in_it;
    ++out_it;
  }
  // At this point, the input domain should have been scanned
  // entirely.
  TORCH_INTERNAL_ASSERT(
      in_it == in_root.end(),
      "Unmatched domain detected: ",
      *in_it,
      " of ",
      in_td);
  // On the other hand, the output may still have some domains left,
  // and they must be new broadcast domains.
  for (; out_it != out_root.end(); ++out_it) {
    TORCH_INTERNAL_ASSERT(
        bcast_dim_flags.at(std::distance(out_root.begin(), out_it)),
        "Unmatched domain detected: ",
        *out_it,
        " of ",
        out_td);
    root_map_.new_broadcast_domains_.insert(DomainKey(out_td, *out_it));
  }
}

bool ComputeAtRootDomainMapBuilder::mapAllConsumers(
    const DomainKey& producer_key) {
  auto it = pending_map_.find(producer_key);
  if (it == pending_map_.end()) {
    return false;
  }
  const auto& consumer_set = it->second;
  // All entries in key_set must be equivalent with each other.
  TORCH_INTERNAL_ASSERT(consumer_set.size() > 0);
  bool consistent = safeToMap(consumer_set);
  if (consistent) {
    for (const auto pending_consumer : consumer_set) {
      setMapped(producer_key, pending_consumer);
    }
  }
  // This entry should never be used again, so remove it.
  pending_map_.erase(it);
  return consistent;
}

void ComputeAtRootDomainMapBuilder::handle(TensorView* tv) {
  const TensorDomain* td = tv->domain();
  const auto root = TensorDomain::noReductions(td->getMaybeRFactorDomain());
  for (auto id : root) {
    if (root_map_.hasConcretizedDomains(td, id)) {
      for (const auto& key : root_map_.getConcretizedKeys(td, id)) {
        mapAllConsumers(key);
      }
    } else {
      mapAllConsumers(DomainKey(td, id));
    }
  }
}

// Checks whether all consumers of a producer can be joined without
// introducing unsupported mappings. Specifically, if a domain of a
// consumer has a mapped iteration domain in another consumer that
// does not correspond to the same producer iteration domain, mapping
// the consumer domains would result in the producer iteration domain
// mapped to two different consumer iteration domains, requiring
// recomputations.
bool ComputeAtRootDomainMapBuilder::hasMatchingDomains(
    const std::vector<DomainKey>& unique_domains) {
  for (const auto& key : unique_domains) {
    for (const auto& other_key : unique_domains) {
      if (key == other_key) {
        continue;
      }
      const auto& other_root = other_key.td()->getRootDomain();
      if (std::any_of(
              other_root.begin(), other_root.end(), [&](const IterDomain* id) {
                return root_map_.canMap(key, other_key.td(), id);
              })) {
        return true;
      }
    }
  }
  return false;
}

// Checks whether all consumers of a producer can be joined without
// introducing unsupported mappings, i.e., requiring recomputations.
bool ComputeAtRootDomainMapBuilder::safeToMap(const DomainKeySet& domains) {
  if (domains.size() <= 1) {
    return true;
  }
  // Filter out equivalent domains
  std::vector<DomainKey> unique_domains;
  for (const auto& domain : domains) {
    if (std::none_of(
            unique_domains.begin(),
            unique_domains.end(),
            [&](const auto& unique_dom) {
              return root_map_.canMap(domain, unique_dom);
            })) {
      unique_domains.push_back(domain);
    }
  }
  if (hasMatchingDomains(unique_domains)) {
    return false;
  }
  // Can't map if reduction output domains would be mapped
  // if (incompatible_domains_.isReductionOutputMapped(unique_domains,
  // eq_set_)) {
  if (incompatible_domains_.isReductionOutputMapped(
          unique_domains, root_map_)) {
    return false;
  }
  return true;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch

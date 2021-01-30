#include <torch/csrc/jit/codegen/cuda/scheduler_registry.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

bool SchedulerEntry::sameAs(const SchedulerEntry* other) {
  if (has_param_ != other->has_param_) {
    return false;
  }
  if (has_param_) {
    return rparams_ == other->rparams_;
  }
  return true;
}

namespace {
inline bool isTrivialReduction(ReductionOp* red) {
  auto o_tv = red->out()->as<TensorView>();
  // Assuming graph unscheduled at this point.
  for (auto id : o_tv->getRootDomain()) {
    if (id->isReduction() && !id->rawExtent()->isOneInt()) {
      return false;
    }
  }
  return true;
}

std::vector<ReductionOp*> findReductionOps(Fusion* fusion) {
  std::vector<ReductionOp*> red_ops;
  for (auto expr : fusion->exprs()) {
    if (auto red = expr->as<ReductionOp>()) {
      if (!isTrivialReduction(red)) {
        red_ops.push_back(red);
      }
    }
  }
  return red_ops;
}

std::vector<TensorView*> findOutputsOfRed(Fusion* fusion, TensorView* red_tv) {
  TORCH_INTERNAL_ASSERT(fusion->inFusion(red_tv));
  auto output_set = DependencyCheck::getAllOutputsOf({red_tv});
  auto tv_entries = ir_utils::filterByType<TensorView>(output_set);
  std::vector<TensorView*> tv_outputs_of_reduction(
      tv_entries.begin(), tv_entries.end());
  return tv_outputs_of_reduction;
}

} // namespace

namespace {

class SingleReductionScheduler : public SchedulerEntry {
 public:
  explicit SingleReductionScheduler(Fusion* fusion, ExpressionEvaluator& ee)
      : SchedulerEntry(ScheduleHeuristic::Reduction, true) {
    getHeuristics(fusion, ee);
  }

  //! Check if the reduction heuristics apply in given fusion
  static bool canSchedule(Fusion* fusion) {
    auto red_ops = findReductionOps(fusion);
    if (red_ops.size() != 1) {
      return false;
    }

    auto red_tv = red_ops[0]->out()->as<TensorView>();

    // Not allowing broadcasting reduction result to support
    //  grid reduction. This is an overkill might want to consider
    //  trying to get the heuristics and check only if grid reduction is
    //  required.
    auto uses = DependencyCheck::getAllUseChains(red_tv);
    for (auto& chain : uses) {
      for (auto val : chain) {
        if (val->definition()->isA<BroadcastOp>()) {
          return false;
        }
      }
    }

    return true;
  }

  void schedule(Fusion* fusion) override {
    // TODO find outputs of tv: what would we need to fill in?
    auto red_tv = findReductionTV(fusion);
    auto output_tv = findOutputsOfRed(fusion, red_tv);
    scheduleReduction(fusion, rparams_, red_tv, output_tv);
  }

 private:
  void getHeuristics(Fusion* fusion, ExpressionEvaluator& ee) {
    auto red_tv = findReductionTV(fusion);
    auto param = getReductionHeuristics(fusion, ee, red_tv);
    TORCH_INTERNAL_ASSERT(param.has_value());
    rparams_ = param.value();
  }

  TensorView* findReductionTV(Fusion* fusion) {
    for (auto expr : fusion->exprs()) {
      if (auto red = expr->as<ReductionOp>()) {
        if (!isTrivialReduction(red)) {
          return red->out()->as<TensorView>();
        }
      }
    }
    TORCH_INTERNAL_ASSERT(false, "unreachable");
    return nullptr;
  }
};

class PointWiseScheduler : public SchedulerEntry {
 public:
  explicit PointWiseScheduler(Fusion* fusion)
      : SchedulerEntry(ScheduleHeuristic::PointWise, false) {}

  static bool canSchedule(Fusion* fusion) {
    auto red_ops = findReductionOps(fusion);
    return red_ops.empty();
  }

  void schedule(Fusion* fusion) override {
    scheduleFusion(fusion);
  }
};

namespace {

// duplicated from Benchmark/utils.h
static void analyzeFusion(
    Fusion* fusion,
    std::vector<TensorView*>& reduction_tv,
    std::vector<TensorView*>& other_tv) {
  auto all_values = DependencyCheck::getAllValsBetween(
      {fusion->inputs().begin(), fusion->inputs().end()}, fusion->outputs());

  for (auto tv : ir_utils::filterByType<TensorView>(all_values)) {
    if (tv->hasReduction()) {
      reduction_tv.push_back(tv);
    } else if (!fusion->hasInput(tv)) {
      other_tv.push_back(tv);
    }
  }
}

} // namespace
class NormalizationScheduler : public SchedulerEntry {
 public:
  explicit NormalizationScheduler(Fusion* fusion, ExpressionEvaluator& ee)
      : SchedulerEntry(ScheduleHeuristic::Normalization, true) {
    getHeuristics(fusion, ee);
  }

  void schedule(Fusion* fusion) override {
    std::vector<TensorView*> reduction_tensors;
    std::vector<TensorView*> other_tensors;
    analyzeFusion(fusion, reduction_tensors, other_tensors);
    scheduleNormalization(fusion, rparams_, reduction_tensors, other_tensors);
  }

  static bool canSchedule(Fusion* fusion) {
    auto red_ops = findReductionOps(fusion);

    if (red_ops.size() < 2) {
      // Use single reduction or pointwise logic
      return false;
    }
    // Before examining the reduction axes want to quickly
    //   check the reductions have the same axis width
    //   to avoid building root domain map in easier cases
    int axis_count = -1;
    auto reduction_root = [](ReductionOp* rop) {
      return rop->out()->as<TensorView>()->getRootDomain();
    };
    for (auto red : red_ops) {
      if (axis_count == -1) {
        axis_count = reduction_root(red).size();
      } else {
        if (reduction_root(red).size() != axis_count) {
          return false;
        }
      }
    }

    // Use root domain map to check the reduction ops have the same axes
    FusionGuard fg(fusion);
    ComputeAtRootDomainMap root_map;
    root_map.build(true);

    // red_ops.size()>1 checked before
    for (size_t it = 1; it < red_ops.size(); it++) {
      if (!checkEquivalence(red_ops[0], red_ops[1], root_map)) {
        return false;
      }
    }
    return true;
  }

 private:
  void getHeuristics(Fusion* fusion, ExpressionEvaluator& ee) {
    std::vector<TensorView*> red_tvs;
    for (auto red : findReductionOps(fusion)) {
      red_tvs.push_back(red->out()->as<TensorView>());
    }
    auto rparams = getNormalizationHeuristics(fusion, ee, red_tvs);
    TORCH_INTERNAL_ASSERT(rparams.has_value());
    rparams_ = rparams.value();
  }

  static inline bool checkEquivalence(
      ReductionOp* op0,
      ReductionOp* op1,
      const ComputeAtRootDomainMap& root_map) {
    const auto out_tv0 = op0->out()->as<TensorView>();
    const auto out_tv1 = op1->out()->as<TensorView>();
    const auto out_root0 = out_tv0->getRootDomain();
    const auto out_root1 = out_tv1->getRootDomain();
    const auto domain0 = out_tv0->domain();
    const auto domain1 = out_tv1->domain();

    TORCH_INTERNAL_ASSERT(out_root0.size() == out_root1.size());
    for (size_t it = 0; it < out_root0.size(); it++) {
      if (!root_map.canMap(domain0, out_root0[it], domain1, out_root1[it])) {
        return false;
      }
    }
    return true;
  }
};

// Schedule Table
const static std::vector<ScheduleHeuristic>& all_heuristics() {
  static const std::vector<ScheduleHeuristic> hlist = {
      ScheduleHeuristic::Reduction,
      ScheduleHeuristic::PointWise,
      ScheduleHeuristic::Normalization};
  return hlist;
}

// Simple dispatcher interface
bool canSchedule(ScheduleHeuristic sh, Fusion* fusion) {
  switch (sh) {
    case ScheduleHeuristic::PointWise:
      return PointWiseScheduler::canSchedule(fusion);
    case ScheduleHeuristic::Reduction:
      return SingleReductionScheduler::canSchedule(fusion);
    case ScheduleHeuristic::Normalization:
      return NormalizationScheduler::canSchedule(fusion);
    default:
      TORCH_INTERNAL_ASSERT(false, "unreachable");
      return false;
  }
  return false;
}
} // namespace

std::unique_ptr<SchedulerEntry> SchedulerEntry::makeEntry(
    ScheduleHeuristic sh,
    Fusion* fusion,
    ExpressionEvaluator& ee) {
  switch (sh) {
    case ScheduleHeuristic::PointWise:
      return std::make_unique<PointWiseScheduler>(fusion);
    case ScheduleHeuristic::Reduction:
      return std::make_unique<SingleReductionScheduler>(fusion, ee);
    case ScheduleHeuristic::Normalization:
      return std::make_unique<NormalizationScheduler>(fusion, ee);
    default:
      TORCH_INTERNAL_ASSERT(false, "unreachable");
  }
  return nullptr;
}

// Simply loop through the list as baseline strategy
c10::optional<ScheduleHeuristic> SchedulerEntry::proposeHeuristics(
    Fusion* fusion) {
  for (auto sh : all_heuristics()) {
    if (canSchedule(sh, fusion)) {
      return sh;
    }
  }
  return c10::nullopt;
}

size_t SchedulerEntryHash::operator()(const SchedulerEntry& se) const {
  if (!se.hasParam()) {
    return 1;
  } else {
    return ReductionParamsHash()(se.params());
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
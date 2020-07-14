#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_base_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>

/*
 * Nodes in here should generally not be used by users. They should be behind
 * the scenes and users shouldn't have to be aware of what they do to use the
 * code generator.
 */

namespace torch {
namespace jit {
namespace fuser {

// Returns true if both v1 and v2 are scalars, are the same type of scalars, and
// dispatches to the inherited Val type's `->sameAs` call. e.g. if both vals are
// `Int` will dispatch to v1->as<Int>()->sameAs(v2.as<Int>())
bool areEqualScalars(Val* v1, Val* v2);

/*
 * TODO: improve implementation bool IterDomain::sameAs(const IterDomain*) const
 * TODO: Add testing of sameAs functions for these nodes
 */

/*
 * A specialization for Unary operations. Unary operations take in a single
 * input and produce a single output. Examples include:
 *   1) Casting operation i.e. float(a_val)
 *   2) Negation i.e. val * -1
 *   3) Reduction across a dimension i.e. val.sum(axis=2)
 *   4) split/merge
 */
class TORCH_CUDA_API UnaryOp : public Expr {
 public:
  ~UnaryOp() = default;
  UnaryOp(UnaryOpType _type, Val* _out, Val* _in);

  UnaryOp(const UnaryOp* src, IrCloner* ir_cloner);

  UnaryOp(const UnaryOp& other) = delete;
  UnaryOp& operator=(const UnaryOp& other) = delete;

  UnaryOp(UnaryOp&& other) = delete;
  UnaryOp& operator=(UnaryOp&& other) = delete;

  Val* out() const {
    return out_;
  }
  Val* in() const {
    return in_;
  }

  UnaryOpType getUnaryOpType() const {
    return unary_op_type_;
  }

  bool sameAs(const UnaryOp* const other) const;

 private:
  const UnaryOpType unary_op_type_;
  Val* const out_ = nullptr;
  Val* const in_ = nullptr;
};

/*
 * A specialization for Binary operations. Binary operations take in two inputs
 * and produce a single output. Examples include:
 *  1) Add/mul/div/mod/sub (A * B)
 *  2) LT (A < B)
 */
class TORCH_CUDA_API BinaryOp : public Expr {
 public:
  ~BinaryOp() = default;
  BinaryOp(BinaryOpType _type, Val* _out, Val* _lhs, Val* _rhs);

  BinaryOp(const BinaryOp* src, IrCloner* ir_cloner);

  BinaryOp(const BinaryOp& other) = delete;
  BinaryOp& operator=(const BinaryOp& other) = delete;

  BinaryOp(BinaryOp&& other) = delete;
  BinaryOp& operator=(BinaryOp&& other) = delete;

  Val* out() const {
    return out_;
  }
  Val* lhs() const {
    return lhs_;
  }
  Val* rhs() const {
    return rhs_;
  }

  BinaryOpType getBinaryOpType() const {
    return binary_op_type_;
  }

  bool sameAs(const BinaryOp* other) const;

 private:
  const BinaryOpType binary_op_type_;
  Val* const out_ = nullptr;
  Val* const lhs_ = nullptr;
  Val* const rhs_ = nullptr;
};

/*
 * Broadcast _in to match _out. broadcast_dims are relative to out. Where
 * broadcast_dims.size() + _in->nDims() == _out->nDims().
 */
class TORCH_CUDA_API BroadcastOp : public Expr {
 public:
  ~BroadcastOp() = default;
  BroadcastOp(Val* _out, Val* _in);

  BroadcastOp(const BroadcastOp* src, IrCloner* ir_cloner);

  BroadcastOp(const BroadcastOp& other) = delete;
  BroadcastOp& operator=(const BroadcastOp& other) = delete;

  BroadcastOp(BroadcastOp&& other) = delete;
  BroadcastOp& operator=(BroadcastOp&& other) = delete;

  Val* out() const {
    return out_;
  }
  Val* in() const {
    return in_;
  }

  bool sameAs(const BroadcastOp* const other) const;

 private:
  Val* const out_ = nullptr;
  Val* const in_ = nullptr;
};

/*
 * Reduction operatoin. Out is first initialized to _init. Then
 * _reduction_op_type is used to update out as out = reductionOp(out, in).
 * Output's axes marked as reduction will be reduced to produce an output
 * tensor. The output tensors size will be the size of all
 * non-reduction/non-broadcast dimensions.
 */
class TORCH_CUDA_API ReductionOp : public Expr {
 public:
  ~ReductionOp() = default;
  ReductionOp(BinaryOpType _reduction_op_type, Val* _init, Val* _out, Val* _in);

  ReductionOp(const ReductionOp* src, IrCloner* ir_cloner);

  ReductionOp(const ReductionOp& other) = delete;
  ReductionOp& operator=(const ReductionOp& other) = delete;

  ReductionOp(ReductionOp&& other) = delete;
  ReductionOp& operator=(ReductionOp&& other) = delete;

  Val* out() const {
    return out_;
  }
  Val* in() const {
    return in_;
  }
  Val* init() const {
    return init_;
  }

  BinaryOpType getReductionOpType() const {
    return reduction_op_type_;
  }

  bool sameAs(const ReductionOp* const other) const;

  std::vector<IterDomain*> getReductionDomains() const;

  std::unordered_map<ParallelType, IterDomain*, TypeHash>
  getParallelReductionDomains() const;

 private:
  const BinaryOpType reduction_op_type_;
  Val* const init_ = nullptr;
  Val* const out_ = nullptr;
  Val* const in_ = nullptr;
};

/*
 * Reduction operatoin. Out is first initialized to _init. Then
 * _reduction_op_type is used to update out as out = reductionOp(out, in).
 * Output's axes marked as reduction will be reduced to produce an output
 * tensor. The output tensors size will be the size of all
 * non-reduction/non-broadcast dimensions.
 */
class TORCH_CUDA_API GridReduction : public Expr {
 public:
  ~GridReduction() = default;
  GridReduction(ReductionOp* _reduction_op);
  GridReduction(
      ReductionOp* _reduction_op,
      Allocate* _reduction_buffer,
      Allocate* _sync_buffer);

  GridReduction(const GridReduction& other) = delete;
  GridReduction& operator=(const GridReduction& other) = delete;

  GridReduction(GridReduction&& other) = delete;
  GridReduction& operator=(GridReduction&& other) = delete;

  ReductionOp* reduction_op() const {
    return reduction_op_;
  }
  Allocate* reduction_buffer() const {
    return reduction_buffer_;
  }
  Allocate* sync_buffer() const {
    return sync_buffer_;
  }

  bool sameAs(const GridReduction* other) const;

 private:
  ReductionOp* reduction_op_ = nullptr;
  Allocate* reduction_buffer_ = nullptr;
  Allocate* sync_buffer_ = nullptr;
};

class TORCH_CUDA_API TernaryOp : public Expr {
 public:
  ~TernaryOp() = default;
  TernaryOp(TernaryOpType _type, Val* _out, Val* _in1, Val* _in2, Val* _in3);

  TernaryOp(const TernaryOp* src, IrCloner* ir_cloner);

  TernaryOp(const TernaryOp& other) = delete;
  TernaryOp& operator=(const TernaryOp& other) = delete;

  TernaryOp(TernaryOp&& other) = delete;
  TernaryOp& operator=(TernaryOp&& other) = delete;

  Val* out() const {
    return out_;
  }

  Val* in1() const {
    return in1_;
  }
  Val* in2() const {
    return in2_;
  }
  Val* in3() const {
    return in3_;
  }

  TernaryOpType getTernaryOpType() const {
    return ternary_op_type_;
  }

  bool sameAs(const TernaryOp* other) const;

 private:
  const TernaryOpType ternary_op_type_;
  Val* const out_ = nullptr;
  Val* const in1_ = nullptr;
  Val* const in2_ = nullptr;
  Val* const in3_ = nullptr;
};

/*
 * Simply a representation of an annotated 1D iterable from start to extent.
 * TensorDomains which represent how to iterate over a tensor is made up of
 * IterDomains to form an ND iterable. We directly set parallization strategies
 * on IterDomains.
 */
class TORCH_CUDA_API IterDomain : public Val {
 public:
  ~IterDomain() = default;

  IterDomain() = delete;

  IterDomain(
      Val* _start,
      Val* _extent,
      ParallelType _parallel_method = ParallelType::Serial,
      bool _reduction_domain = false,
      bool _rfactor_domain = false,
      bool _broadcast_domain = false);

  IterDomain(const IterDomain* src, IrCloner* ir_cloner);

  bool sameAs(const IterDomain* const other) const;

  // Returns a new IterDomain matching properties of this
  IterDomain* clone() const {
    return new IterDomain(
        start(),
        extent(),
        parallel_method(),
        isReduction(),
        isRFactorProduct(),
        isBroadcast());
  }

  static IterDomain* merge(IterDomain* outer, IterDomain* inner);

  // TODO: Make protected and friend TensorDomain so only it can call into this
  // directly, users should not be able to use this call
  static std::pair<IterDomain*, IterDomain*> split(IterDomain* in, Val* factor);

  bool isReduction() const {
    return is_reduction_domain_;
  }

  bool isRFactorProduct() const {
    return is_rfactor_domain_;
  }

  bool isBroadcast() const {
    return is_broadcast_domain_;
  }

  bool isParallelized() const {
    return parallel_method_ != ParallelType::Serial;
  }

  // Return if this iter domain is mapped to a grid dimension
  bool isBlockDim() const {
    return (
        parallel_method_ == ParallelType::BIDz ||
        parallel_method_ == ParallelType::BIDy ||
        parallel_method_ == ParallelType::BIDx);
  }

  // Return if this iter domain is mapped to a block dimension
  bool isThreadDim() const {
    return (
        parallel_method_ == ParallelType::TIDz ||
        parallel_method_ == ParallelType::TIDy ||
        parallel_method_ == ParallelType::TIDx);
  }

  // Return if this iter domain is either mapped to a block or grid dimension
  bool isThread() const {
    return (isBlockDim() || isThreadDim());
  }

  void parallelize(ParallelType t) {
    parallel_method_ = t;

    TORCH_CHECK(
        t != ParallelType::Vectorize, "Vectorization not yet supported.");

    if (t == ParallelType::Unroll)
      TORCH_CHECK(
          start()->isZeroInt() && extent()->isConstScalar(),
          "Unrolling only supported with start = 0 and extent as a const int, but got ",
          "a start of ",
          start(),
          " and extent ",
          extent(),
          " .");
  }

  ParallelType parallel_method() const {
    return parallel_method_;
  }

  Val* start() const {
    return start_;
  }
  Val* extent() const;
  Val* rawExtent() const {
    return extent_;
  }

  IterDomain(const IterDomain& other) = delete;
  IterDomain& operator=(const IterDomain& other) = delete;

  IterDomain(IterDomain&& other) = delete;
  IterDomain& operator=(IterDomain&& other) = delete;

 private:
  Val* const start_ = nullptr;
  Val* const extent_ = nullptr;
  ParallelType parallel_method_ = ParallelType::Serial;
  bool is_reduction_domain_ = false;
  bool is_rfactor_domain_ = false;
  bool is_broadcast_domain_ = false;
};

/*
 * TensorDomain holds a vector of IterDomains. It holds an IterDomain for every
 * logical axis in its associated tensor. TensorDomain does not directly hold
 * the Tensor it is associated with, and in theory could be associated with
 * multiple tensors. TensorDomain's primary responsibility is to provide a
 * mechanism to access history of transformations that were used to generate it.
 * This is done through the normal interaction of Expr/Val in Fusion. i.e. if we
 * want to know the previous operation generating a particular TensorDomain we
 * can simply call FusionGuard::getCurFusion()->origin(a_tensor_domain) which
 * should give us an operation in the list [split, merge] or similar
 * operations that take in a TensorDomain, applies a transformation and outputs
 * a tensor domain.
 */
class TORCH_CUDA_API TensorDomain : public Val {
 public:
  TensorDomain() = delete;
  ~TensorDomain() = default;

  TensorDomain(const TensorDomain& other) = delete;
  TensorDomain& operator=(const TensorDomain& other) = delete;

  TensorDomain(TensorDomain&& other) = delete;
  TensorDomain& operator=(TensorDomain&& other) = delete;

  explicit TensorDomain(std::vector<IterDomain*> _domain);

  TensorDomain(
      std::vector<IterDomain*> _root_domain,
      std::vector<IterDomain*> _domain);

  TensorDomain(
      std::vector<IterDomain*> _root_domain,
      std::vector<IterDomain*> _rfactor_domain,
      std::vector<IterDomain*> _domain);

  TensorDomain(const TensorDomain* src, IrCloner* ir_cloner);

  std::vector<IterDomain*>::size_type nDims() const {
    return domain_.size();
  }

  bool sameAs(const TensorDomain* const other) const;

  static bool sameAs(
      const std::vector<IterDomain*>& lhs,
      const std::vector<IterDomain*>& rhs);

  const std::vector<IterDomain*>& domain() const {
    return domain_;
  }

  bool hasReduction() const;
  bool hasBlockReduction() const;
  bool hasGridReduction() const;
  bool hasBroadcast() const;
  bool hasRFactor() const;

  const std::vector<IterDomain*>& noReductions() const {
    return no_reduction_domain_;
  }

  const std::vector<IterDomain*>& noBroadcasts() const {
    return no_bcast_domain_;
  }

  const std::vector<IterDomain*>& rootDomain() const {
    return root_domain_;
  };

  const std::vector<IterDomain*>& rfactorDomain() const {
    return rfactor_domain_;
  };

  void resetDomains() {
    no_reduction_domain_ = noReductions(domain_);
    no_bcast_domain_ = noBroadcasts(domain_);
  }

  // i here is int, as we want to accept negative value and ::size_type can be a
  // uint.
  IterDomain* axis(int i) const;

  size_t posOf(IterDomain* id) const;

  // Split "axis" into 2 axes where the inner axes is size of "factor"
  // and outer axis is size axis.size() / factor. Allow factor to be symbolic
  // value instead of constant.
  // TODO: Make protected and friend TensorDomain so only it can call into this
  // directly, users should not be able to use this call
  void split(int axis_, Val* factor);

  // Merge axis_o and axis_i. axis_i is the fast changing dimension. Resulting
  // axis is by default placed at original position axis_o
  void merge(int axis_o, int axis_i);

  // Reorder axes according to map[old_pos] = new_pos
  void reorder(const std::unordered_map<int, int>& old2new);

  static std::vector<IterDomain*> orderedAs(
      const std::vector<IterDomain*>& td,
      const std::unordered_map<int, int>& old2new);

  static std::vector<IterDomain*> noReductions(const std::vector<IterDomain*>&);
  static std::vector<IterDomain*> noBroadcasts(const std::vector<IterDomain*>&);

  static bool hasBroadcast(const std::vector<IterDomain*>&);
  static bool hasReduction(const std::vector<IterDomain*>&);

  // pair is in order where second is the consumer of first
  std::pair<TensorDomain*, TensorDomain*> rFactor(const std::vector<int>& axes);

 private:
  const std::vector<IterDomain*> root_domain_;
  std::vector<IterDomain*> domain_;
  std::vector<IterDomain*> no_bcast_domain_;
  std::vector<IterDomain*> no_reduction_domain_;
  const std::vector<IterDomain*> rfactor_domain_;
};

/*
 * Representation a split on an IterDomain by "factor"
 * TODO: Implement split by nparts
 */
class TORCH_CUDA_API Split : public Expr {
 public:
  ~Split() = default;

  Split(const Split& other) = delete;
  Split& operator=(const Split& other) = delete;

  Split(Split&& other) = delete;
  Split& operator=(Split&& other) = delete;

  Split(IterDomain* _outer, IterDomain* _inner, IterDomain* _in, Val* _factor);

  Split(const Split* src, IrCloner* ir_cloner);

  IterDomain* outer() const {
    return outer_;
  }
  IterDomain* inner() const {
    return inner_;
  }
  IterDomain* in() const {
    return in_;
  }
  Val* factor() const {
    return factor_;
  }
  bool sameAs(const Split* const other) const;

 private:
  IterDomain* const outer_ = nullptr;
  IterDomain* const inner_ = nullptr;
  IterDomain* const in_ = nullptr;
  Val* const factor_ = nullptr;
};

/*
 * Merge the IterDomains outer and inner into one domain, outer and inner
 * dictate which will be traversed first (inner). Both IterDomains must be of
 * the same iter or reduction type, as well as the same parallelization strategy
 * if there is one.
 * TODO: Should this be a unary op type?
 */
class TORCH_CUDA_API Merge : public Expr {
 public:
  ~Merge() = default;
  Merge(IterDomain* _out, IterDomain* _outer, IterDomain* _inner);

  Merge(const Merge* src, IrCloner* ir_cloner);

  Merge(const Merge& other) = delete;
  Merge& operator=(const Merge& other) = delete;

  Merge(Merge&& other) = delete;
  Merge& operator=(Merge&& other) = delete;

  IterDomain* out() const {
    return out_;
  }
  IterDomain* outer() const {
    return outer_;
  }
  IterDomain* inner() const {
    return inner_;
  }

  bool sameAs(const Merge* const other) const;

 private:
  IterDomain* const out_ = nullptr;
  IterDomain* const outer_ = nullptr;
  IterDomain* const inner_ = nullptr;
};

/*
 * ForLoop provides scoping around an int iterator from 0 to range. Exprs placed
 * in its body are considered inside the scope of the for loop. In the future
 * the implementation should look quite different so that we can do proper
 * dependency annalysis like in Fusion.
 *
 * TODO: Change implmentation of Exprs contained in the scope to be more similar
 * to Fusion where we can do proper dependency analysis.
 */
class TORCH_CUDA_API ForLoop : public Expr {
 public:
  ~ForLoop() = default;
  ForLoop(
      Val* _index,
      IterDomain* _iter_domain,
      const std::vector<Expr*>& _body = {},
      Expr* parent_scope = nullptr);

  ForLoop(const ForLoop* src, IrCloner* ir_cloner);

  ForLoop(const ForLoop& other) = delete;
  ForLoop& operator=(const ForLoop& other) = delete;

  ForLoop(ForLoop&& other) = delete;
  ForLoop& operator=(ForLoop&& other) = delete;

  Val* index() const {
    return index_;
  }

  IterDomain* iter_domain() const {
    return iter_domain_;
  }

  Scope& body() {
    return body_;
  }

  const Scope& constBody() const {
    return body_;
  }

  bool sameAs(const ForLoop* other) const;
  Expr* parentScope() const {
    return parent_scope_;
  }

 private:
  Val* const index_ = nullptr;
  IterDomain* const iter_domain_;
  Scope body_;
  Expr* parent_scope_ = nullptr;
};

/*
 * IfThenElse provides scoping for an boolean operator. Exprs placed in its body
 * are considered inside the scope of the if statement. In the future the
 * implementation should look quite different so that we can do proper
 * dependency annalysis like in Fusion.
 *
 * TODO: Change implmentation of Exprs contained in the scope to be more similar
 * to Fusion where we can do proper dependency analysis.
 */
class TORCH_CUDA_API IfThenElse : public Expr {
 public:
  ~IfThenElse() = default;
  IfThenElse(
      Bool* _cond,
      const std::vector<Expr*>& _if_body = {},
      const std::vector<Expr*>& _else_body = {},
      Expr* _parent_scope = nullptr);

  IfThenElse(const IfThenElse* src, IrCloner* ir_cloner);

  IfThenElse(const IfThenElse& other) = delete;
  IfThenElse& operator=(const IfThenElse& other) = delete;

  IfThenElse(IfThenElse&& other) = delete;
  IfThenElse& operator=(IfThenElse&& other) = delete;

  Bool* cond() const {
    return cond_;
  }

  const Scope& constBody() const {
    return body_;
  }

  const Scope& constElseBody() const {
    return else_body_;
  }

  Scope& body() {
    return body_;
  }

  Scope& elseBody() {
    return else_body_;
  }

  bool hasElse() const {
    return !else_body_.empty();
  }

  bool sameAs(const IfThenElse* other) const;

  Expr* parentScope() const {
    return parent_scope_;
  }

 private:
  Bool* const cond_ = nullptr;
  Scope body_;
  Scope else_body_;
  Expr* parent_scope_ = nullptr;
};

/*
 * TODO: Fill out TensorIndex, which is a list of Ints used to directly index a
 * TensorView. It is not the flattened index, which needs to be computed using
 * stride information.
 */
class TORCH_CUDA_API TensorIndex : public Val {
 public:
  ~TensorIndex() = default;

  TensorIndex(const TensorIndex& other) = delete;
  TensorIndex& operator=(const TensorIndex& other) = delete;

  TensorIndex(TensorIndex&& other) = delete;
  TensorIndex& operator=(TensorIndex&& other) = delete;

  TensorIndex(const TensorView* const _view, std::vector<Val*> _indices)
      : Val(ValType::TensorIndex, _view->getDataType().value()),
        view_(_view),
        indices_(_indices) {
    TORCH_INTERNAL_ASSERT(
        std::all_of(
            _indices.begin(),
            _indices.end(),
            [](Val* v) {
              return (v->getValType() == ValType::Scalar ||
                      v->getValType() == ValType::NamedScalar) &&
                  v->getDataType() == DataType::Int;
            }),
        "Cannot index with a value other than an int.");
  }

  TensorIndex(const TensorIndex* src, IrCloner* ir_cloner);

  std::vector<Val*>::size_type nDims() const {
    return indices_.size();
  }

  // i here is int, as we want to accept negative value and ::size_type can be a
  // uint.
  Val* index(int i) const;

  const std::vector<Val*>& indices() const {
    return indices_;
  }

  const TensorView* view() const {
    return view_;
  }

  bool sameAs(const TensorIndex* const other) const;

 private:
  const TensorView* view_ = nullptr;
  std::vector<Val*> indices_;
};

/*
 * Allocate is a lower level Node that describes a buffer of memory that
 * is required as an intermediate within a kernel.  The extent is the expression
 * of the size of the buffer that is generated from the TensorView that
 * describes the output of an operation.
 *
 * TODO: The components of Allocate like Type and Name could be separated from
 * the the assocated TensorView.  Perhaps that is more appropriate?
 */
class TORCH_CUDA_API Allocate : public Expr {
 public:
  ~Allocate() = default;

  Allocate(const Allocate& other) = delete;
  Allocate& operator=(const Allocate& other) = delete;

  Allocate(Allocate&& other) = delete;
  Allocate& operator=(Allocate&& other) = delete;

  Allocate(
      Val* _buffer,
      MemoryType _memory_type = MemoryType::Local,
      Val* _size = nullptr);

  Allocate(const Allocate* src, IrCloner* ir_cloner);

  Val* buffer() const {
    return buffer_;
  }

  MemoryType getMemoryType() const {
    return memory_type_;
  }

  Val* size() const {
    return size_;
  }

  DataType buffer_type() const {
    return buffer_->getDataType().value();
  }

  bool sameAs(const Allocate* other) const;

 private:
  Val* buffer_ = nullptr;
  MemoryType memory_type_ = MemoryType::Local;
  Val* size_ = nullptr;
};

/*
 * Integer value which has a special name. These could be:
 * - threadIdx.x
 * - blockIdx.y
 * - blockDim.z
 * - T3.stride[2]
 */
class TORCH_CUDA_API NamedScalar : public Val {
 public:
  ~NamedScalar() = default;
  NamedScalar() = delete;

  NamedScalar(std::string _name, DataType dtype)
      : Val(ValType::NamedScalar, dtype), name_(_name) {}

  NamedScalar(const NamedScalar* src, IrCloner* ir_cloner);

  NamedScalar(const NamedScalar& other) = delete;
  NamedScalar& operator=(const NamedScalar& other) = delete;

  NamedScalar(NamedScalar&& other) = delete;
  NamedScalar& operator=(NamedScalar&& other) = delete;

  const std::string& name() const {
    return name_;
  }

  bool sameAs(const NamedScalar* const other) const {
    return other->name().compare(name()) == 0;
  }

  // Return the named scalar extent of a parallel dimension (e.g. blockDim.x)
  static NamedScalar* getParallelDim(ParallelType p_type);

  // Return the named scalar index of a parallel dimension (e.g. threadIdx.x)
  static NamedScalar* getParallelIndex(ParallelType p_type);

  // Return the parallel type of this NamedScalar if it is an extent of a
  // parallel dimension
  c10::optional<ParallelType> getParallelDim() const;

  // Return the parallel type of this NamedScalar if it is an index of a
  // parallel dimension
  c10::optional<ParallelType> getParallelIndex() const;

 private:
  std::string name_;
};

} // namespace fuser
} // namespace jit
} // namespace torch

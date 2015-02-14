#ifndef SILL_ARRAY_FACTOR_HPP
#define SILL_ARRAY_FACTOR_HPP

#include <sill/base/bounded_domain.hpp>
#include <sill/base/finite_assignment.hpp>
#include <sill/datastructure/finite_index.hpp>
#include <sill/factor/base/factor.hpp>
#include <sill/functional/assign.hpp>
#include <sill/functional/operators.hpp>

#include <Eigen/Core>

#include <stdexcept>

namespace sill {

  /**
   * A base class for discrete factors with up to two arguments. This
   * class stores the parameters of the factor as an Eigen array
   * and provides standard indexing functions join/aggregate/restrict
   * functions on the factors. This class does not model the Factor
   * concept.
   *
   * \tparam T the type of parameters stored in the table.
   * \see canonical_array, probability_array
   */
  template <typename T>
  class array_factor : public factor {
  public:
    // Underlying representation
    typedef Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> array_type;

    // Range types
    typedef T*       iterator;
    typedef const T* const_iterator;
    typedef T        value_type;

    // Domain
    typedef bounded_domain<finite_variable*, 2> domain_type;

    // Constructors
    //==========================================================================

    /**
     * Default constructor. Creates an empty factor.
     */
    array_factor() { }

    /**
     * Constructs an array_factor with the given arguments and initializes its
     * parameters to the given array.
     *
     * \param zero_nan if true, any NaN parameters will be zeroed out
     */
    explicit array_factor(const domain_type& args,
                          const array_type& param,
                          bool zero_nan)
      : args_(args), param_(param) {
      check_param();
      if (zero_nan) clear_nan();
    }

    /**
     * Constructs an array_factor with the given arguments and moves its
     * parameters from the given array.
     *
     * \param zero_nan if true, any NaN parameters will be zeroed out
     */
    explicit array_factor(const domain_type& args,
                          array_type&& param,
                          bool zero_nan)
      : args_(args) {
      param_.swap(param);
      check_param();
      if (zero_nan) clear_nan();
    }

    //! Copy constructor.
    array_factor(const array_factor& other) = default;

    //! Move constructor.
    array_factor(array_factor&& other) {
      swap(other);
    }

    //! Assignment operator.
    array_factor& operator=(const array_factor& other) = default;

    //! Move assignment operator.
    array_factor& operator=(array_factor&& other) {
      swap(other);
      return *this;
    }

    // Serialization and initialization
    //==========================================================================
    
    //! Serializes members.
    void save(oarchive& ar) const {
      //ar << args_ << param_;
    }

    //! Deserializes members.
    void load(iarchive& ar) {
      //ar >> args_ >> param_;
      check_param();
    }

    /**
     * Resets the content of this factor to the given arguments.
     * The array elements may be invalidated.
     */
    void reset(const domain_type& args) {
      if (args_ != args || empty()) {
        args_ = args;
        param_.resize(x() ? x()->size() : 1, y() ? y()->size() : 1);
      }
    }

    /**
     * Resets the content of this factor to an empty domain.
     * The array elements may be invalidated.
     */
    void reset() {
      args_.clear();
      param_.resize(1, 1);
    }

    // Accessors
    //==========================================================================

    //! Returns the arguments of this factor.
    const domain_type& arguments() const {
      return args_;
    }

    //! Returns the first argument or NULL if the factor is empty or nullary.
    finite_variable* x() const {
      return args_[0];
    }

    //! Returns the second argument or NULL if the factor has <=1 arguments.
    finite_variable* y() const {
      return args_[1];
    }

    //! Returns the number of arguments of this factor.
    size_t arity() const {
      return args_.size();
    }

    //! Returns the total number of elements of the factor.
    size_t size() const {
      return param_.size();
    }

    //! Returns true if the factor is empty (equivalent to size() == 0).
    bool empty() const {
      return !param_.data();
    }

    //! Returns the pointer to the first element (undefined if the factor is empty).
    T* begin() {
      return param_.data();
    }

    //! Returns the pointer to the first element (undefined if the factor is empty).
    const T* begin() const {
      return param_.data();
    }

    //! Returns the pointer past the last element (undefined if the factor is empty).
    T* end() {
      return param_.data() + param_.size();
    }

    //! Returns the pointer past the last element (undefined if the factor is empty).
    const T* end() const {
      return param_.data() + param_.size();
    }

    //! Returns the parameter with the given linear index.
    T& operator[](size_t i) {
      return param_(i);
    }

    //! Returns the parameter with the given linear index.
    const T& operator[](size_t i) const {
      return param_(i);
    }

    //! Provides mutable access to the parameter array of this factor.
    array_type& param() {
      return param_;
    }

    //! Returns the parameter array of this factor.
    const array_type& param() const {
      return param_;
    }

    //! Returns the parameter for the given assignment.
    T& param(const finite_assignment& a) {
      return param_(linear_index(a));
    }

    //! Returns the parameter for the given assignment.
    const T& param(const finite_assignment& a) const {
      return param_(linear_index(a));
    }

    //! Returns the parameter for the given index.
    T& param(const finite_index& index) {
      return param_(linear_index(index));
    }

    //! Returns the parameter of rthe given index.
    const T& param(const finite_index& index) const {
      return param_(linear_index(index));
    }

    // Indexing
    //==========================================================================

    /**
     * Converts a linear index to the corresponding assignment to the
     * factor arguments.
     */
    void assignment(size_t linear_index, finite_assignment& a) const {
      if (x()) { a[x()] = linear_index % param_.rows(); }
      if (y()) { a[y()] = linear_index / param_.rows(); }
    }

    /**
     * Returns the linear index corresponding to the given assignment.
     */
    size_t linear_index(const finite_assignment& a) const {
      if (x() && y()) {
        return safe_get(a, x()) + safe_get(a, y()) * param_.rows();
      } else {
        return x() ? safe_get(a, x()) : 0;
      }
    }

    /**
     * Returns the linear index corresponding to the given finite index.
     */
    size_t linear_index(const finite_index& index) const {
      switch (index.size()) {
      case 0: return 0;
      case 1: return index[0];
      case 2: return index[0] + index[1] * param_.rows();
      default: throw std::invalid_argument(
          "An index with >2 elements passed to a array factor"
        );
      }
    }

    /**
     * Substitutes this factor's arguments according to the given map
     * in place.
     */
    void subst_args(const finite_var_map& var_map) {
      for (finite_variable*& x : args_) {
        finite_variable* xn = safe_get(var_map, x);
        if (!x->type_compatible(xn)) {
          throw std::invalid_argument(
            "subst_args: " + x->str() + " and " + xn->str() + " are not compatible"
          );
        }
        x = xn;
      }
    }

    /**
     * Checks if ths dimensions of the parameter array match the factor
     * arguments.
     * \throw std::runtime_error if some of the dimensions do not match
     */
    void check_param() const {
      if (param_.rows() != (x() ? x()->size() : 1)) {
        throw std::runtime_error("Invalid number of rows");
      }
      if (param_.cols() != (y() ? y()->size() : 1)) {
        throw std::runtime_error("Invalid number of columns");
      }
    }

    /**
     * Replaces NaNs with 0s. This operation is used when applying division
     * for probability_array factors.
     */
    void clear_nan() {
      for (T& value : *this) {
        if (std::isnan(value)) { value = T(0); }
      }
    }
    
    // Implementations of common factor operations
    //========================================================================
  protected:
    /**
     * Joins this factor in place with f using the given mutating operation.
     * f must not introduce any new arguments into this factor.
     */
    template <typename Op>
    void join_inplace(const array_factor& f, Op op, bool zero_nan) {
      switch (f.arity()) {
      case 0: // combine with a constant element-wise
        op(param_, f[0]);
        break;
      case 1: // combine with a vector column- or row-wise
        if (f.x() == x()) {
          op(param_.colwise(), f.param_.col(0));
        } else if (f.x() == y()) {
          op(param_.rowwise(), f.param_.col(0).transpose());
        } else {
          throw std::invalid_argument(
            "array_factor: inplace operation introduces new variable " + f.x()->str()
          );
        }
        break;
      case 2: // combine with a array directly or via transpose
        if (x() == f.x() && y() == f.y()) {
          op(param_, f.param_);
        } else if (x() == f.y() && y() == f.x()) {
          //op(param_, f.param_.transpose());
          op(param_, f.param_.transpose());
        } else {
          throw std::invalid_argument(
            "array_factor: inplace operation introduces new variable(s)"
          );
        }
        break;
      }

      // convert nans back to zeros after division by zero
      if (zero_nan) { clear_nan(); }
    }

    /**
     * Transforms and aggregates the parameter array of this factor along
     * all dimensions other than those for the retained variables and
     * stores the result to the specified factor. When none of the
     * dimensions are eliminated, copies the parameters (possibly transposed).
     */
    template <typename TransOp, typename AggOp>
    void transform_aggregate(const domain_type& retained,
                             TransOp trans_op,
                             AggOp agg_op,
                             array_factor& result) const {
      switch (retained.size()) {
      case 0: // aggregate all elements
        result.reset();
        result[0] = agg_op(trans_op(param_));
        return;
      case 1: // aggregate to one dimension
        result.reset(retained);
        if (result.x() == x()) {
          result.param_ = agg_op(trans_op(param_).rowwise());
        } else if (result.x() == y()) {
          result.param_ = agg_op(trans_op(param_).colwise()).transpose();
        } else {
          throw std::invalid_argument(
            "array_factor: retained variable not in the factor domain"
          );
        }
        return;
      case 2: // no aggregation; just copy the result
        if (retained == args_) {
          result = *this;
        } else if (retained[0] == y() && retained[1] == x()) {
          result.reset(retained);
          result.param_ = param_.transpose();
        } else {
          throw std::invalid_argument(
            "array_factor: retained variables not in the factor domain"
          );
        }
        return;
      }
      throw std::logic_error("array_factor: invalid arity in aggregate");
    }

    /**
     * Aggregates the parameter array of this factor along
     * all dimensions other than those for the retained variables and
     * stores the result to the specified factor. When none of the
     * dimensions are eliminated, copies the parameters (possibly transposed).
     * Shortcut for transform_aggregate<identity, AggOp>.
     */
    template <typename AggOp>
    void aggregate(const domain_type& retained, AggOp agg_op,
                   array_factor& result) const {
      transform_aggregate(retained, identity(), agg_op, result);
    }

    /**
     * Restricts this factor to an assignment and stores the result to the
     * given array factor. This function must be protected, because the
     * result is not strongly typed, i.e., we could accidentally restrict
     * a probability_array and store the result in a canonical_array or
     * vice versa.
     */
    void restrict(const finite_assignment& a, array_factor& result) const {
      switch (arity()) {
      case 0:   // this factor is already a constant; nothing to restrict
        result = *this;
        return;
      case 1: { // if the assignment contains x(), restrict; otherwise not
        auto itx = a.find(x());
        if (itx != a.end()) {
          result.reset();
          result[0] = param_(itx->second);
        } else {
          result = *this;
        }
        return;
      }
      case 2: { // restrict both, one, or neither argument
        auto itx = a.find(x());
        auto ity = a.find(y());
        if (itx != a.end() && ity != a.end()) {
          result.reset();
          result[0] = param_(itx->second, ity->second);
        } else if (itx != a.end()) {
          result.reset({y()});
          result.param_ = param_.row(itx->second).transpose();
        } else if (ity != a.end()) {
          result.reset({x()});
          result.param_ = param_.col(ity->second);
        } else {
          result = *this;
        }
        return;
      }
      }
      throw std::logic_error("array_factor: invalid arity in restrict");
    }

    /**
     * Restricts this factor to an assignment and joins the result to the
     * given result factor using the given mutating operation. The join
     * operation must not introduce any new arguments to the result.
     */
    template <typename Op>
    void restrict_join(const finite_assignment& a, Op op, bool zero_nan,
                       array_factor& result) const {
      bool rx = a.count(x());
      bool ry = a.count(y());
      if (!rx && !ry) { // nothing to restrict, just join
        result.join_inplace(*this, op, zero_nan);
        return;
      }

      switch (arity()) {
      case 1:            // restricting x()
        op(result.param_, param(a));
        break;
      case 2:
        if (rx && ry) {  // restricting both x() and y()
          op(result.param_, param(a));
        } else if (rx) { // restricting x(); join in a row
          size_t i = safe_get(a, x());
          if (y() == result.x()) {
            op(result.param_.colwise(), param_.row(i).transpose());
          } else if (y() == result.y()) {
            op(result.param_.rowwise(), param_.row(i));
          } else {
            throw std::invalid_argument(
              "restrict_join introduces " + y()->str() + " into the result"
            );
          }
        } else if (ry) { // restricting y(); join in a column
          size_t i = safe_get(a, y());
          if (x() == result.x()) {
            op(result.param_.colwise(), param_.col(i));
          } else if (result.y() == x()) {
            op(result.param_.rowwise(), param_.col(i).transpose());
          } else {
            throw std::invalid_argument(
              "restrict_join introduces " + x()->str() + " into the result"
            );
          }
        }
        break;
      default:
        throw std::logic_error("restrict_join: inconsistent state");
      }

      // convert nans back to zeros after division by zero
      if (zero_nan) { result.clear_nan(); }
    }

    /**
     * Transforms the elements and aggregates them using the given operation.
     */
    template <typename TransOp, typename AccuOp>
    T transform_accumulate(T init,  TransOp trans_op, AccuOp accu_op) const { 
      T result(init);
      for (const T& x : *this) {
        result = accu_op(result, trans_op(x));
      }
      return result;
    }

  protected:
    /**
     * Implementation of the swap function. This function must be protected,
     * because it's not type-safe.
     */
    void swap(array_factor& other) {
      if (this != &other) {
        using std::swap;
        swap(args_, other.args_);
        param_.swap(other.param_);
      }
    }

    /**
     * Implementation of operator==(). This function must be protected,
     * because it's not type-safe.
     */
    bool equal(const array_factor& other) const {
      return args_ == other.args_ && std::equal(begin(), end(), other.begin());
    }

    // Protected data members
    //========================================================================

    //! The arguments of this factor.
    domain_type args_;

    //! The parameter array.
    array_type param_;

  }; // class array_factor

  // Utility functions
  //========================================================================

  /**
   * Throws an std::invalid_argument exception if the two factors do not
   * have the same argument vectors.
   */
  template <typename T>
  void check_same_arguments(const array_factor<T>& f, const array_factor<T>& g) {
    if (f.arguments() != g.arguments()) {
      throw std::invalid_argument(
        "Element-wise operations require the two factors to have the same arguments"
      );
    }
  }

  /**
   * Joins the parameter tables of two factors using a binary operation.
   * The resulting factor contains the union of f's and g's argument sets.
   * This operation is only supported if the result has at most two arguments.
   */
  template <typename Result, typename T, typename Op>
  Result join(const array_factor<T>& f,
              const array_factor<T>& g,
              Op op, bool zero_nan = false) {
    typedef typename array_factor<T>::array_type array_type;
    using Eigen::Replicate;
    using Eigen::Dynamic;

    const array_type& a = f.param();
    const array_type& b = g.param();

    size_t nf = f.arity();
    size_t ng = g.arity();
    
    if (nf == 0) { // combine a constant with all elements of g
      return Result(g.arguments(), op(a(0), b), zero_nan);
    }
    if (ng == 0) { // combine all elements of f with a constant
      return Result(f.arguments(), op(a, b(0)), zero_nan);
    }
    if (nf == 1 && ng == 1) {
      if (f.x() == g.x()) { // direct combination
        return Result({f.x()}, op(a.col(0), b.col(0)), zero_nan);
      } else {              // outer combination
        Result result({f.x(), g.x()});
        T* r = result.begin();
        for (size_t j = 0; j < size_t(b.rows()); ++j) {
          T y = b(j);
          const T* x = a.data();
          for (size_t i = 0; i < size_t(a.rows()); ++i) {
            *r++ = op(*x++, y);
          }
        }
        if (zero_nan) result.clear_nan();
        return result;
      }
    }
    if (nf == 1 && ng == 2) {
      if (f.x() == g.x()) { // combine f with each column of g
        Replicate<decltype(a.col(0)), 1, Dynamic> arep(a.col(0), 1, b.cols());
        return Result({g.x(), g.y()}, op(arep, b), zero_nan);
      }
      if (f.x() == g.y()) { // combine f with each row of g 
        Replicate<decltype(a.col(0)), 1, Dynamic> arep(a.col(0), 1, b.rows());
        return Result({g.x(), g.y()}, op(arep, b.transpose()), zero_nan);
      }
      throw std::invalid_argument("array_factor: join creates a ternary factor");
    }
    if (nf == 2 && ng == 1) {
      if (f.x() == g.x()) { // combine each column of f with g
        Replicate<decltype(b.col(0)), 1, Dynamic> brep(b.col(0), 1, a.cols());
        return Result(f.arguments(), op(a, brep), zero_nan);
      }
      if (f.y() == g.x()) { // combine each row of f with g
        return Result(f.arguments(), op(a.rowwise(), b.col(0).transpose()), zero_nan);
      }
      throw std::invalid_argument("array_factor: join creates a ternary factor");
    }
    if (nf == 2 && ng == 2) {
      if (f.x() == g.x() && f.y() == g.y()) { // direct combination
        return Result(f.arguments(), op(a, b), zero_nan);
      }
      if (f.x() == g.y() && f.y() == g.x()) { // combination with a transpose
        return Result(f.arguments(), op(a, b.transpose()), zero_nan);
      }
      throw std::invalid_argument("array_factor: join creates a 3/4-ary factor");
    }
    throw std::invalid_argument("array_factor: Invalid arity of inputs in join");
  }

  /**
   * Transforms the parameters of two factors using a binary operation
   * and returns the result. The two factors must have the same domains.
   */
  template <typename Result, typename T, typename Op>
  Result transform(const array_factor<T>& f, const array_factor<T>& g, Op op) {
    check_same_arguments(f, g);
    Result result(f.arguments());
    std::transform(f.begin(), f.end(), g.begin(), result.begin(), op);
    return result;
  }

  /**
   * Transforms the parameters of two factors using a binary operation
   * and accumulates the result using another operation.
   */
  template <typename T, typename JoinOp, typename AggOp>
  T transform_accumulate(const array_factor<T>& f,
                         const array_factor<T>& g,
                         JoinOp join_op,
                         AggOp agg_op) {
    assert(f.arguments() == g.arguments());
    return std::inner_product(f.begin(), f.end(), g.begin(), T(0), agg_op, join_op);
  }

  /*
  //! Product-marginal
  void product_marginal(f, g, retain, result) { }
  */

} // namespace sill

#endif

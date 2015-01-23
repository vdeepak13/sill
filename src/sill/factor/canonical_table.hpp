#ifndef SILL_CANONICAL_TABLE_HPP
#define SILL_CANONICAL_TABLE_HPP

#include <sill/global.hpp>
#include <sill/base/finite_assignment.hpp>
#include <sill/datastructure/table.hpp>
#include <sill/factor/factor.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/factor/traits.hpp>
#include <sill/functional/operators.hpp>
#include <sill/functional/entropy.hpp>
#include <sill/math/constants.hpp>
#include <sill/math/logarithmic.hpp>
#include <sill/range/algorithm.hpp>
#include <sill/serialization/serialize.hpp>

#include <initializer_list>
#include <iostream>
#include <map>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A factor of a categorical probability distribution represented in the
   * canonical form of the exponential family. This factor represents a
   * non-negative funciton over finite variables X as f(X | \theta) =
   * exp(\sum_x \theta_x * 1(X=x)). In some cases, e.g. in a Bayesian network,
   * this factor also represents a probability distribution in the log-space.
   *
   * \tparam T a real type for representing each parameter
   *
   * \ingroup factor_types
   * \see Factor
   */
  template <typename T = double>
  class canonical_table : public factor {
  public: 
    // Public types
    //==========================================================================

    // Factor member types
    typedef T                 real_type;
    typedef logarithmic<T>    result_type;
    typedef finite_variable   variable_type;
    typedef finite_domain     domain_type;
    typedef finite_var_vector var_vector_type;
    typedef finite_assignment assignment_type;
    struct param_type; // forward declaration
    
    // IndexableFactor member types
    typedef finite_index index_type;
    
    // DistributionFactor member types
    typedef boost::function<canonical_table(const finite_domain&)>
      marginal_fn_type;
    typedef boost::function<canonical_table(const finite_domain&,
                                            const finite_domain&)>
      conditional_fn_type;
    typedef table_factor probability_factor_type;
    
    // LearnableFactor member types
    typedef finite_dataset dataset_type;
    typedef finite_record_old record_type;

    // Range types
    typedef T* iterator;
    typedef const T* const_iterator;
    typedef T value_type;

    // Constructors and conversion operators
    //==========================================================================

    //! Default constructor. Creates an empty factor.
    canonical_table() { }

    //! Constructs a factor with given arguments and uninitialized parameters.
    explicit canonical_table(const finite_var_vector& args) {
      reset(args);
    }

    //! Constructs a factor equivalent to a constant.
    explicit canonical_table(logarithmic<T> value) {
      reset(finite_var_vector());
      param_[0] = value.lv;
    }

    //! Constructs a factor with the given arguments and constant likelihood.
    canonical_table(const finite_var_vector& args, logarithmic<T> value) {
      reset(args);
      param_.fill(value.lv);
    }

    //! Constructs a factor with the given argument set and constant likelihood.
    canonical_table(const finite_domain& args, logarithmic<T> value) {
      reset(make_vector(args));
      param_.fill(value.lv);
    }

    //! Creates a factor with the specified arguments and parameters.
    canonical_table(const finite_var_vector& args, const param_type& param)
      : args_(args.begin(), args.end()),
        arg_vec_(args),
        param_(param) {
      check_param();
    }

    //! Creates a factor with the specified arguments and parameters.
    canonical_table(const finite_var_vector& args, std::initializer_list<T> values) {
      reset(args);
      assert(values.size() == size());
      std::copy(values.begin(), values.end(), begin());
    }

    //! Conversion from a table_factor.
    explicit canonical_table(const table_factor& f) {
      *this = f;
    }

    //! Assigns a constant to this factor.
    canonical_table& operator=(logarithmic<T> value) {
      reset(finite_var_vector());
      param_[0] = value.lv;
      return *this;
    }

    //! Assigns a probability table factor to this factor.
    canonical_table& operator=(const table_factor& f) {
      reset(f.arg_vector());
      std::transform(f.begin(), f.end(), begin(), logarithm<T>());
      return *this;
    }

    //! Exchanges the content of two factors.
    friend void swap(canonical_table& f, canonical_table& g) {
      if (&f != &g) {
        using std::swap;
        swap(f.args_, g.args_);
        swap(f.arg_vec_, g.arg_vec_);
        swap(f.param_, g.param_);
      }
    }

    //! Serializes members.
    void save(oarchive & ar) const {
      ar << arg_vec_ << param_;
    }

    //! Deserializes members.
    void load(iarchive & ar) {
      ar >> arg_vec_ >> param_;
      args_.clear();
      args_.insert(arg_vec_.begin(), arg_vec_.end());
      check_param();
    }

    /**
     * Resets the content of this factor to the given sequence of arguments.
     * If the table size changes, the table elements become invalidated.
     */
    void reset(const finite_var_vector& args) {
      if (empty() || arg_vec_ != args) {
        arg_vec_ = args;
        args_.clear();
        args_.insert(args.begin(), args.end());
        finite_index shape(args.size());
        for (size_t i = 0; i < args.size(); ++i) {
          shape[i] = args[i]->size();
        }
        param_.reset(shape);
      }
    }

    // Accessors and comparison operators
    //==========================================================================

    //! Returns the argument set of this factor.
    const finite_domain& arguments() const {
      return args_;
    }

    //! Returns the argument vector of this factor.
    const finite_var_vector& arg_vector() const {
      return arg_vec_;
    }

    //! Returns the number of arguments of this factor.
    size_t arity() const {
      return param_.arity();
    }

    //! Returns the total number of elements of the factor.
    size_t size() const {
      return param_.size();
    }

    //! Returns true if the factor has an empty table (equivalent to size() == 0).
    bool empty() const {
      return param_.empty();
    }

    //! Returns the pointer to the first element or NULL if the factor is empty.
    T* begin() {
      return param_.begin();
    }

    //! Returns the pointer to the first element or NULL if the factor is empty.
    const T* begin() const {
      return param_.begin();
    }

    //! Returns the pointer to past the last element or NULL if the factor is empty.
    T* end() {
      return param_.end();
    }

    //! Returns the pointer to past the last element or NULL if the factor is empty.
    const T* end() const {
      return param_.end();
    }

    //! Returns the parameter with the given linear index.
    const T& operator[](size_t i) const {
      return param_[i];
    }

    //! Provides mutable access to the parameter with the given linear index.
    T& operator[](size_t i) {
      return param_[i];
    }
    
    //! Returns the parameters of this factor.
    const param_type& param() const { 
      return param_;
    }

    //! Provides mutable access to the parameters of this factor.
    param_type& param() {
      return param_;
    }

    //! Returns the parameter for the given assignment.
    const T& param(const finite_assignment& a) const {
      return param_[index(a)];
    }

    //! Provides mutable access to the paramater for the given assignment.
    T& param(const finite_assignment& a) {
      return param_[index(a)];
    }
    
    //! Returns the parameter for the given index.
    const T& param(const finite_index& index) const {
      return param_(index);
    }

    //! Provides mutable access to the parameter for the given index.
    T& param(const finite_index& index) {
      return param_(index);
    }

    //! Returns the value of the factor for the given assignment.
    logarithmic<T> operator()(const finite_assignment& a) const {
      return logarithmic<T>(param_[index(a)], log_tag());
    }

    //! Returns the value of the factor for the given index.
    logarithmic<T> operator()(const finite_index& index) const {
      return logarithmic<T>(param_(index), log_tag());
    }

    //! Returns the log-value of the factor for the given assignment.
    T log(const finite_assignment& a) const {
      return param_[index(a)];
    }

    //! Returns the log-value of the factor for the given index.
    T log(const finite_index& index) const {
      return param_(index);
    }

    //! Returns true if the two factors have the same argument vectors and values.
    bool operator==(const canonical_table& other) const {
      return arg_vec_ == other.arg_vec_ && param_ == other.param_;
    }

    //! Returns true if the two factors do not have the same arguments or values.
    bool operator!=(const canonical_table& other) const {
      return !operator==(other);
    }

    // Indexing
    //==========================================================================
    
    /**
     * Converts the index to this factor's arguments to an assignment.
     * The index may be merely a prefix, and the ouptut assignment is not cleared.
     */
    void assignment(const finite_index& index, finite_assignment& a) const {
      assert(index.size() <= arg_vec_.size());
      for(size_t i = 0; i < index.size(); i++) { a[arg_vec_[i]] = index[i]; }
    }

    /**
     * Returns the linear index corresponding to the given assignment.
     * If strict, each argument of this factor must be present in the
     * assignment. If not strict, the missing arguments will be associated
     * with value 0.
     */
    size_t index(const finite_assignment& a, bool strict = true) const {
      size_t result = 0;
      for (size_t i = 0; i < arity(); ++i) {
        finite_variable* v = arg_vec_[i];
        finite_assignment::const_iterator it = a.find(v);
        if (it != a.end()) {
          result += param_.offset().multiplier(i) * it->second;
        } else if (strict) {
          throw std::invalid_argument(
            "The assignment does not contain the variable " + v->str()
          );
        }
      }
      return result;
    }

    /**
     * Returns the mapping of this factor's arguments to the given var vector.
     * If strict, all the arguments must be present in the given vector.
     * If not strict, the missing variables will be assigned a NA value,
     * std::numeric_limits<size_t>::max().
     *
     * When using this function in factor operations, always call the
     * dim_map function on the factor whose elements will be iterated
     * over in a non-linear fashion. The vector vars are the arguments
     * of the table that is iterated over in a linear fashion.
     */
    finite_index dim_map(const finite_var_vector& vars, bool strict = true) const {
      finite_index map(arity(), std::numeric_limits<size_t>::max());
      for(size_t i = 0; i < map.size(); i++) {
        finite_var_vector::const_iterator it = 
          std::find(vars.begin(), vars.end(), arg_vec_[i]);
        if (it != vars.end()) {
          map[i] = it - vars.begin();
        } else if (strict) {
          throw std::invalid_argument("Missing variable " + arg_vec_[i]->str());
        }
      }
      return map;
    }

    /**
     * Substitutes this factor's arguments according to the given map,
     * in place.
     */
    void subst_args(const finite_var_map& var_map) {
      args_ = subst_vars(args_, var_map);
      foreach (finite_variable*& var, arg_vec_) {
        finite_variable* new_var = safe_get(var_map, var);
        if (!var->type_compatible(new_var)) {
          throw std::invalid_argument(
            "subst_args: " + var->str() + " and " + new_var->str() +
            " are not compatible"
          );
        }
        var = new_var;
      }
    }

    /**
     * Checks if the shape of the table matches this factor's argument vector.
     * \throw std::runtime_error if some of the dimensions do not match
     */
    void check_param() const {
      if (param_.arity() != arg_vec_.size()) {
        throw std::runtime_error("Invalid table arity");
      }
      for (size_t i = 0; i < arg_vec_.size(); ++i) {
        if (param_.size(i) != arg_vec_[i]->size()) {
          throw std::runtime_error("Invalid table shape");
        }
      }
    }

    // Factor operations
    //==========================================================================
    
    //! Multiplies another factor into this one.
    canonical_table& operator*=(const canonical_table& f) {
      return join_inplace(f, std::plus<T>());
    }

    //! Divides another factor into this one.
    canonical_table& operator/=(const canonical_table& f) {
      return join_inplace(f, std::minus<T>());
    }

    //! Multiplies this factor by a constant.
    canonical_table& operator*=(logarithmic<T> x) {
      param_.transform(incremented_by<T>(x.lv));
      return *this;
    }

    //! Divides this factor by a constant.
    canonical_table& operator/=(logarithmic<T> x) {
      param_.transform(decremented_by<T>(x.lv));
      return *this;
    }

    //! Multiplies two canonical_table factors.
    friend canonical_table
    operator*(const canonical_table& f, const canonical_table& g) {
      return join(f, g, std::plus<T>());
    }

    //! Divides two canonical_table factors.
    friend canonical_table
    operator/(const canonical_table& f, const canonical_table& g) {
      return join(f, g, std::minus<T>());
    }

    //! Multiplies a canonical_table factor by a constant.
    friend canonical_table
    operator*(const canonical_table& f, logarithmic<T> x) {
      return transform(f, incremented_by<T>(x.lv));
    }

    //! Multiplies a canonical_table factor by a constant.
    friend canonical_table
    operator*(logarithmic<T> x, const canonical_table& f) {
      return transform(f, incremented_by<T>(x.lv));
    }

    //! Divides a canonical_table factor by a constant.
    friend canonical_table
    operator/(const canonical_table& f, logarithmic<T> x) {
      return transform(f, decremented_by<T>(x.lv));
    }

    //! Divides a constant by a canonical_table factor.
    friend canonical_table
    operator/(logarithmic<T> x, const canonical_table& f) {
      return transform(f, subtracted_from<T>(x.lv));
    }

    //! Raises the canonical_table factor by an exponent.
    friend canonical_table
    pow(const canonical_table& f, T x) {
      return transform(f, multiplied_by<T>(x));
    }

    //! Returns the sum of the probabilities of two factors.
    friend canonical_table
    operator+(const canonical_table& f, const canonical_table& g) {
      return transform(f, g, log_sum_exp<T>());
    }

    //! Element-wise maximum of two factors.
    friend canonical_table
    max(const canonical_table& f, const canonical_table& g) {
      return transform(f, g, sill::maximum<T>());
    }
  
    //! Element-wise minimum of two factors.
    friend canonical_table
    min(const canonical_table& f, const canonical_table& g) {
      return transform(f, g, sill::minimum<T>());
    }

    //! Returns \f$f^{(1-a)} * g^a\f$.
    friend canonical_table
    weighted_update(const canonical_table& f, const canonical_table& g, T a) {
      return transform(f, g, sill::weighted_plus<T>(1 - a, a));
    }

    //! Computes the marginal of the factor over a subset of variables.
    canonical_table marginal(const finite_domain& retain) const {
      canonical_table result;
      marginal(retain, result);
      return result;
    }

    //! Computes the maximum for each assignment to the given variables.
    canonical_table maximum(const finite_domain& retain) const {
      return aggregate(retain, -inf<T>(), sill::maximum<T>());
    }

    //! Computes the minimum for each assignment to the given variables.
    canonical_table minimum(const finite_domain& retain) const {
      return aggregate(retain, +inf<T>(), sill::minimum<T>());
    }

    //! If this factor represents p(x, y), returns p(x | y).
    canonical_table conditional(const finite_domain& tail) const {
      return (*this) / marginal(tail);
    }

    //! Computes the marginal of the factor over a subset of variables.
    void marginal(const finite_domain& retain, canonical_table& result) const {
      T offset = param_.max();
      aggregate(retain, T(0), plus_exp<T>(-offset), result);
      foreach(T& x, result.param_) { x = std::log(x) + offset; }
    }

    //! Computes the maximum for each assignment to the given variables.
    void maximum(const finite_domain& retain, canonical_table& result) const {
      aggregate(retain, -inf<T>(), sill::maximum<T>(), result);
    }

    //! Computes the minimum for each assignment to the given variables.
    void minimum(const finite_domain& retain, canonical_table& result) const {
      aggregate(retain, +inf<T>(), sill::minimum<T>(), result);
    }

    //! Returns the normalization constant of the factor.
    logarithmic<T> marginal() const {
      T offset = param_.max();
      T sum = param_.accumulate(T(0), plus_exp<T>(-offset));
      return logarithmic<T>(std::log(sum) + offset, log_tag());
    }

    //! Returns the maximum value in the factor.
    logarithmic<T> maximum() const {
      return logarithmic<T>(param_.max(), log_tag());
    }

    //! Returns the minimum value in the factor.
    logarithmic<T> minimum() const {
      return logarithmic<T>(param_.min(), log_tag());
    }

    //! Computes the maximum value and stores the corresponding assignment.
    logarithmic<T> maximum(finite_assignment& a) const {
      const T* it = std::max_element(begin(), end());
      assignment(param_.index(it), a);
      return logarithmic<T>(*it, log_tag());
    }

    //! Computes the minimum value and stores the corresponding assignment.
    logarithmic<T> minimum(finite_assignment& a) const {
      const T* it = std::min_element(begin(), end());
      assignment(param_.index(it), a);
      return logarithmic<T>(*it, log_tag());
    }

    //! Normalizes the factor in-place.
    canonical_table& normalize() {
      param_ -= marginal().lv;
      return *this;
    }

    //! Returns true if the factor is normalizable (approximation).
    bool is_normalizable() const {
      return boost::math::isfinite(param_.max());
    }

    //! Restricts this factor to an assignment.
    canonical_table restrict(const finite_assignment& a) const {
      canonical_table result;
      restrict(a, result);
      return result;
    }

    //! Restricts this factor to an assignment.
    void restrict(const finite_assignment& a, canonical_table& result) const {
      finite_var_vector new_vars;
      foreach (finite_variable* v, arg_vec_) {
        if (!a.count(v)) { new_vars.push_back(v); }
      }
      result.reset(new_vars);
      if (prefix(result.arg_vec_, arg_vec_)) {
        result.param_.restrict(param_, index(a, false));
      } else {
        finite_index map = dim_map(result.arg_vec_, false);
        table_restrict<T>(result.param_, param_, map, index(a, false))();
      }
    }

    // Entropy and divergences
    //==========================================================================
    
    //! Computes the entropy for the distribution represented by this factor.
    T entropy() const {
      return param_.transform_accumulate(T(0), entropy_log_op<T>(), std::plus<T>());
    }

    //! Computes the entropy for a subset of variables. Performs marginalization.
    T entropy(const finite_domain& a) const {
      return (args_ == a) ? entropy() : marginal(a).entropy();
    }

    //! Computes the mutual information between two subsets of this factor's
    //! arguments.
    T mutual_information(const finite_domain& a, const finite_domain& b) const {
      return entropy(a) + entropy(b) - entropy(set_union(a, b));
    }

    //! Computes the cross entropy from p to q.
    friend T cross_entropy(const canonical_table& p, const canonical_table& q) {
      return transform_accumulate(p, q, entropy_log_op<T>(), std::plus<T>());
    }

    //! Computes the Kullback-Liebler divergence from p to q.
    friend T kl_divergence(const canonical_table& p, const canonical_table& q) {
      return transform_accumulate(p, q, kld_log_op<T>(), std::plus<T>());
    }

    //! Computes the Jensen–Shannon divergece between p and q.
    friend T js_divergence(const canonical_table& p, const canonical_table& q) {
      return transform_accumulate(p, q, jsd_log_op<T>(), std::plus<T>());
    }

    //! Computes the sum of absolute differences between the parameters of p and q.
    friend T sum_diff(const canonical_table& p, const canonical_table& q) {
      return transform_accumulate(p, q, abs_difference<T>(), std::plus<T>());
    }
    
    //! Computes the max of absolute differences between the parameters of p and q.
    friend T max_diff(const canonical_table& p, const canonical_table& q) {
      return transform_accumulate(p, q, abs_difference<T>(), sill::maximum<T>());
    }

    // Join, transform, and aggregate operations
    //==========================================================================

    /**
     * Joins this factor in place with f using the given binary operation.
     * f must not introduce any new arguments into this factor.
     */
    template <typename Op>
    canonical_table& join_inplace(const canonical_table& f, Op op) {
      if (arg_vec_ == f.arg_vec_) {
        param_.transform(f.param_, op);
      } else {
        finite_index f_map = f.dim_map(arg_vec_);
        table_join_inplace<T, T, Op>(param_, f.param_, f_map, op)();
      }
      return *this;
    }
    
    /**
     * Joins the parameter tables of two factors using a binary operation.
     * The resulting factor contains the union of f's and g's argument sets.
     */
    template <typename Op>
    friend canonical_table
    join(const canonical_table& f, const canonical_table& g, Op op) {
      if (f.arg_vec_ == g.arg_vec_) {
        canonical_table result(f.arg_vec_);
        std::transform(f.begin(), f.end(), g.begin(), result.begin(), op);
        return result;
      } else {
        canonical_table result(set_union(f.arg_vec_, g.arg_vec_));
        finite_index f_map = f.dim_map(result.arg_vec_);
        finite_index g_map = g.dim_map(result.arg_vec_);
        table_join<T, T, Op>(result.param_, f.param_, g.param_, f_map, g_map, op)();
        return result;
      }
    }

    /**
     * Aggregates the parameter table of this factor along all dimensions
     * other than those for the retained variables using the given binary
     * operation and initial value. The returned factor contains the retained
     * variables (or a subset thereof) in an unspecified order.
     */
    template <typename Op>
    canonical_table aggregate(const finite_domain& retained, T init, Op op) const {
      canonical_table result;
      aggregate(retained, init, op, result);
      return result;
    }

    /**
     * Aggregates the parameter table of this factor along all dimensions
     * other than those for the retained variables using the given binary
     * operation and stores the result to the specified factor. This function
     * avoids reallocation if the target argument vector has not changed.
     */
    template <typename Op>
    void aggregate(const finite_domain& retained, T init, Op op,
                   canonical_table& result) const {
      finite_domain new_args = set_intersect(args_, retained);
      result.reset(make_vector(new_args));
      result.param_.fill(init);
      finite_index result_map = result.dim_map(arg_vec_);
      table_aggregate<T, T, Op>(result.param_, param_, result_map, op)();
    }

    /**
     * Transforms the parameters of the factor with a unary operation
     * and returns the result.
     */
    template <typename Op>
    friend canonical_table transform(const canonical_table& f, Op op) {
      canonical_table result(f.arg_vec_);
      std::transform(f.begin(), f.end(), result.begin(), op);
      return result;
    }

    /**
     * Transforms the parameters of two factors using a binary operation
     * and returns the result. The two factors must have the same argument vectors.
     */
    template <typename Op>
    friend canonical_table
    transform(const canonical_table& f, const canonical_table& g, Op op) {
      assert(f.arg_vec_ == g.arg_vec_);
      canonical_table result(f.arg_vec_);
      std::transform(f.begin(), f.end(), g.begin(), result.begin(), op);
      return result;
    }

    /**
     * Transforms the parameters of two factors using a binary operation
     * and accumulates the result using another operation.
     */
    template <typename JoinOp, typename AggOp>
    friend T transform_accumulate(const canonical_table& f,
                                  const canonical_table& g,
                                  JoinOp join_op,
                                  AggOp agg_op) {
      assert(f.arg_vec_ == g.arg_vec_);
      return std::inner_product(f.begin(), f.end(), g.begin(), T(0), agg_op, join_op);
    }

    // Parameter vector
    //========================================================================

    /**
     * The type that represents the parameters stored in this factor.
     * Models the OptimizationVector concept.
     */
    struct param_type : public table<T> {
      
      param_type() { }

      explicit param_type(const finite_index& shape)
        : table<T>(shape) { }

      param_type(const finite_index& shape, T init)
        : table<T>(shape, init) { }

      param_type(const finite_index& shape, std::initializer_list<T> values)
        : table<T>(shape, values) { }

      // OptimizationVector functions
      //======================================================================
      param_type& operator+=(const param_type& x) {
        this->transform(x, std::plus<T>());
        return *this;
      }

      param_type& operator-=(const param_type& x) {
        this->transform(x, std::minus<T>());
        return *this;
      }

      param_type& operator+=(T a) {
        this->transform(incremented_by<T>(a));
        return *this;
      }

      param_type& operator-=(T a) {
        this->transform(decremented_by<T>(a));
        return *this;
      }

      param_type& operator*=(T a) {
        this->transform(multiplied_by<T>(a));
        return *this;
      }
      
      param_type& operator/=(T a) {
        this->transform(divided_by<T>(a));
        return *this;
      }

      T max() const {
        return this->accumulate(-inf<T>(), sill::maximum<T>());
      }

      T min() const {
        return this->accumulate(+inf<T>(), sill::minimum<T>());
      }

      friend void axpy(T a, const param_type& x, param_type& y) {
        y.transform(x, sill::plus_multiple<T>(a));
      }
      
      friend T dot(const param_type& x, const param_type& y) {
        assert(x.shape() == y.shape());
        return std::inner_product(x.begin(), x.end(), y.begin(), T(0));
      }

      // LogLikelihoodDerivatives functions
      //====================================================================
      void add_gradient(const param_type& x, const finite_index& index, T w) {
        (*this)(index) += w;
      }

      void add_gradient(const param_type& x, const finite_index& tail,
                        const table_factor& p, T w) {
        assert(p.num_arguments() + tail.size() == this->arity());
        size_t index = this->offset().linear(tail, p.num_arguments());
        for (size_t i = 0; i < p.size(); ++i) {
          (*this)[index + i] += p(i) * w;
        }
      }

      void add_gradient_sqr(const param_type& x, const finite_index& tail,
                            const table_factor& p, T w) {
        add_gradient(x, tail, p, w);
      }

      void add_hessian_diag(const param_type& x, const finite_index& index, T w) { }

      void add_hessian_diag(const param_type& x, const finite_index& index,
                            const table_factor& p, T w) { }

      // ConditionalParameter functions
      //====================================================================
      param_type condition(const finite_index& index) const {
        assert(index.size() <= this->arity());
        size_t n = this->arity() - index.size();
        finite_index shape(this->shape().begin(), this->shape().begin() + n);
        param_type result(shape);
        result.restrict(*this, this->offset().linear(index, n));
        return result;
      }

    }; // struct param_type

    // Private members
    //==========================================================================
  private:
    //! The argument set of this factor.
    finite_domain args_;

    //! The sequence of arguments of this factor
    finite_var_vector arg_vec_;

    //! The canonical parameters of this factor
    param_type param_;

  }; // class canonical_table

  // Input / output
  //============================================================================

  /**
   * Prints a human-readable representatino of the table factor to the stream.
   * \relates canonical_table
   */
  template <typename T>
  std::ostream& operator<<(std::ostream& out, const canonical_table<T>& f) {
    out << "#CT(" << f.arg_vector() << ")" << std::endl;
    out << f.param();
    return out;
  }

  // Utilities - TODO
  //============================================================================


  // Traits
  //============================================================================

  template <typename T>
  struct has_multiplies<canonical_table<T> > : public boost::true_type { };

  template <typename T>
  struct has_multiplies_assign<canonical_table<T> > : public boost::true_type { };

  template <typename T>
  struct has_divides<canonical_table<T> > : public boost::true_type { };

  template <typename T>
  struct has_divides_assign<canonical_table<T> > : public boost::true_type { };

  template <typename T>
  struct has_max<canonical_table<T> > : public boost::true_type { };

  template <typename T>
  struct has_min<canonical_table<T> > : public boost::true_type { };

  template <typename T>
  struct has_marginal<canonical_table<T> > : public boost::true_type { };

  template <typename T>
  struct has_maximum<canonical_table<T> > : public boost::true_type { };

  template <typename T>
  struct has_minimum<canonical_table<T> > : public boost::true_type { };

  template <typename T>
  struct has_arg_max<canonical_table<T> > : public boost::true_type { };

  template <typename T>
  struct has_arg_min<canonical_table<T> > : public boost::true_type { };

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

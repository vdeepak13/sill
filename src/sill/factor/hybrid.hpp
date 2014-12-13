#ifndef SILL_HYBRID_HPP
#define SILL_HYBRID_HPP

#include <sill/base/variable_utils.hpp>
#include <sill/datastructure/dense_table.hpp>
#include <sill/factor/factor.hpp>
#include <sill/factor/util/factor_evaluator.hpp>
#include <sill/factor/util/factor_mle_incremental.hpp>
#include <sill/factor/util/factor_sampler.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/factor/traits.hpp>
#include <sill/functional.hpp>
#include <sill/learning/dataset/hybrid_dataset.hpp>

#include <boost/function.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // note on variable ordering in the components - in the concept?

  /**
   * A class that represents a distribution over discrete and continuous variables.
   * Specifically, it can represent (possibly unnormalized) distributions of the
   * form p(X, Y) = p(X | Y) \times p(Y), where X is vector and Y is finite.
   * Typically, p(X | Y = y) is Gaussian.
   *
   * Models Factor, IndexableFactor, LearnableFactor, DistributionFactor
   *
   * \tparam F The factor that represents each component p(X, y) for one value y.
   *           Typically, either moment_gaussian or canonical_gaussian.
   *           The operations on F must be stable with respect to the ordering
   *           of the variables.
   */
  template <typename F>
  class hybrid {

    // Public types
    //==========================================================================
  public:
    // Factor concept types
    typedef typename F::result_type result_type;
    typedef typename F::real_type   real_type;
    typedef variable                variable_type;
    typedef domain                  domain_type;
    typedef var_vector              var_vector_type;
    typedef assignment              assignment_type;
    
    // IndexableFactor concept types
    typedef hybrid_values<real_type> index_type;

    // DistributionFactor concept types
    typedef boost::function<hybrid(const domain&)> marginal_fn_type;
    typedef boost::function<hybrid(const domain&,
                                   const domain&)> conditional_fn_type;
    // LearnableFactor concept types
    typedef hybrid_dataset<real_type> dataset_type;
    typedef hybrid_record<real_type>  record_type;

    // Range types
    typedef typename dense_table<F>::iterator iterator;
    typedef typename dense_table<F>::const_iterator const_iterator;
    
    // Constructors and assignment operators
    //==========================================================================
  public:
    //! Default constructor for factor with no arguments (i.e., a constant).
    explicit hybrid(result_type value = 1.0) {
      initialize(finite_var_vector(), vector_var_vector(), value);
    }

    //! Constructor for the given finite and vector arguments
    explicit hybrid(const finite_var_vector& finite_args,
                    const vector_var_vector& vector_args = vector_var_vector(),
                    result_type value = 1.0) {
      initialize(finite_args, vector_args, value);
    }

    //! Constructor for the given finite an vector domains
    explicit hybrid(const finite_domain& finite_args,
                    const vector_domain& vector_args = vector_domain(),
                    result_type value = 1.0) {
      initialize(make_vector(finite_args), make_vector(vector_args), value);
    }

    //! Conversion constructor from the table factor
    explicit hybrid(const table_factor& tf) {
      initialize(tf.arg_vector(), vector_var_vector(), 1.0);
      std::copy(tf.begin(), tf.end(), table_.begin());
    }

    //! Conversion constructor from the component factor
    explicit hybrid(const F& f) {
      initialize(finite_var_vector(), f);
    }

    // TODO: Conversion operators to table and component factors 
    
    //! Assignment from another hybrid factor
    hybrid& operator=(const hybrid& other) {
      if (this == &other) {
        return *this;
      }
      if (finite_args_ == other.finite_args_) {
        if (args_ != other.args_) { args_ = other.args_; }
        std::copy(other.begin(), other.end(), table_.begin());
      } else {
        args_ = other.args_;
        finite_args_ = other.finite_args_;
        var_index_ = other.var_index_;
        table_ = other.table_;
      }
      return *this;
    }

    //! Assignment from a constant
    hybrid& operator=(result_type value) {
      initialize(finite_var_vector(), vector_var_vector(), value);
      return *this;
    }

    //! Assignment from a table_factor
    hybrid& operator=(const table_factor& tf) {
      if (finite_args_ != tf.arg_vector()) {
        initialize(tf.arg_vector(), vector_var_vector(), 1.0);
      } else if (args_.size() != finite_args_.size()) {
        args_.clear();
        args_.insert(finite_args_.begin(), finite_args_.end());
      }
      std::copy(tf.begin(), tf.end(), table_.begin());
      return *this;
    }

    //! Assignment from a component factor
    hybrid& operator=(const F& f) {
      initialize(finite_var_vector(), f);
      return *this;
    }

    //! Exchanges the content of two factors
    void swap(hybrid& other) {
      args_.swap(other.args_);
      finite_args_.swap(other.finite_args_);
      var_index_.swap(other.var_index_);
      table_.swap(other.table_);
    }

    //! Updates the argument set after the change in components
    void update_arguments() {
      args_.clear();
      args_.insert(finite_args_.begin(), finite_args_.end());
      args_.insert(begin()->arguments().begin(), begin()->arguments().end());
    }

    // Accessors and evaluators
    //==========================================================================
    //! Returns the argument set of this factor
    const domain& arguments() const {
      return args_;
    }

    //! Returns the arguments of this factor in the internal ordering
    var_vector arg_vector() const {
      return concat(finite_args_, begin()->arg_vector());
    }
    
    //! Returns the finite arguments of this factor in the internal ordering
    const finite_var_vector& finite_args() const {
      return finite_args_;
    }

    //! Returns the arguments of each compoent in the internal ordering
    vector_var_vector vector_args() const {
      return begin()->arg_vector();
    }

    //! Returns the number of finite arguments
    size_t num_finite() const {
      return finite_args_.size();
    }

    //! Returns the number of vector arguments
    size_t num_vector() const {
      return begin()->arguments().size();
    }

    //! Returns an iterator pointing to the first component of this factor
    const_iterator begin() const {
      return table_.begin();
    }

    //! Returns an iterator pointing to the first component of this factor
    iterator begin() {
      return table_.begin();
    }

    //! Returns an iterator pointing to past the last component of this factor
    const_iterator end() const {
      return table_.end();
    }

    //! Returns an iterator pointing to past the last component of this factor
    iterator end() {
      return table_.end();
    }

    //! Returns the number of components (the number of assignments to Y)
    size_t size() const {
      return table_.size();
    }

    //! Returns the underlying table
    const dense_table<F>& table() const {
      return table_;
    }
    
    //! Returns the i-th component of this factor
    const F& operator[](size_t i) const {
      return table_.begin()[i];
    }

    //! Returns the i-th component of this factor
    F& operator[](size_t i) {
      return table_.begin()[i];
    }

    //! Returns the component corresponding to the given finite assignment
    const F& operator()(const finite_assignment& a) const {
      return table_(make_index(a));
    }

    //! Returns the component corresponding to the given finite assignment
    F& operator()(const finite_assignment& a) {
      return table_(make_index(a));
    }

    //! Returns the component corresponding to the given finite index
    const F& operator()(const std::vector<size_t>& index) const {
      return table_(index);
    }

    //! Returns the component corresponding to the given finite index
    F& operator()(const std::vector<size_t>& index) {
      return table_(index);
    }

    //! Returns the value of this factor corresponding to the given assignment
    result_type operator()(const assignment& a) const {
      return table_(make_index(a.finite()))(a.vector());
    }

    //! Returns the value of this factor corresponding to the given assignment
    result_type operator()(const assignment& a) {
      return table_(make_index(a.finite()))(a.vector());
    }

    //! Returns the value of this factor corresponding to the given index
    result_type operator()(const index_type& index) const {
      return table_(index.finite)(index.vector);
    }

    //! Returns true of the factor contains all the given finite variables
    bool contains(const finite_var_vector& vars) const {
      foreach (finite_variable* v, vars) {
        if (!var_index_.count(v)) { return false; }
      }
      return true;
    }

    // Factor operations
    //==========================================================================
    /**
     * Combines two hybrid factors using the given function.
     */
    static hybrid combine(const hybrid& f, const hybrid& g,
                          boost::function<F(const F&, const F&)> op) {
      finite_domain new_finite_args;
      new_finite_args.insert(f.finite_args_.begin(), f.finite_args_.end());
      new_finite_args.insert(g.finite_args_.begin(), g.finite_args_.end());
      hybrid result(new_finite_args);
      result.table_.join(f.table_,
                         g.table_,
                         result.make_dim_map(f.finite_args_),
                         result.make_dim_map(g.finite_args_),
                         op);
      result.args_ = set_union(f.arguments(), g.arguments());
      return result;
    }

    /**
     * Combines a hybrid factor and a table factor using the given function.
     */
    template <typename Op>
    static hybrid combine(const hybrid& f, const table_factor& g, Op op) {
      finite_domain new_finite_args;
      new_finite_args.insert(f.finite_args_.begin(), f.finite_args_.end());
      new_finite_args.insert(g.arg_vector().begin(), g.arg_vector().end());
      hybrid result(new_finite_args);
      result.table_.join(f.table_,
                         g.table(),
                         result.make_dim_map(f.finite_args_),
                         result.make_dim_map(g.arg_vector()),
                         op);
      result.args_ = set_union(f.arguments(), g.arguments());
      return result;
    }

    /**
     * Combines a hybrid factor and a component factor using the given function.
     */
    static hybrid combine(const hybrid& f, const F& g,
                          boost::function<F(const F&, const F&)> op) {
      hybrid result(f.finite_args_);
      for (size_t i = 0; i < f.size(); ++i) {
        result[i] = op(f[i], g);
      }
      result.args_ = set_union(f.arguments(), g.arguments());
      return result;
    }

    /**
     * Combines this factor with a hybrid factor in place using the given
     * function. The finite arguments of f must all be present in *this.
     */
    void combine_in(const hybrid& f, boost::function<F&(F&, const F&)> op) {
      table_.join_with(f.table_, make_dim_map(f.finite_args_), op);
      args_.insert(f.args_.begin(), f.args_.end());
    }

    /**
     * Combines this factor with a table factor in place using the given
     * function. The finite arguments of f must all be present in *this.
     */
    template <typename Op>
    void combine_in(const table_factor& f, Op op) {
      table_.join_with(f.table(), make_dim_map(f.arg_vector()), op);
    }

    /**
     * Combines this factor with a component factor using the given function.
     */
    void combine_in(const F& f, boost::function<F&(F&, const F&)> op) {
      foreach (F& component, table_) { op(component, f); }
      args_.insert(f.arguments().begin(), f.arguments().end());
    }

    /**
     * Combines each component with a constant.
     */
    template <typename Op>
    void combine_in(result_type val, Op op) {
      foreach (F& component, table_) { op(component, val); }
    }

    /**
     * Returns the marginal of this factor over the given set of finite
     * variables.
     */
    table_factor marginal(const finite_domain& retain) const {
      finite_var_vector new_finite_args = intersect(finite_args_, retain);
      table_factor result(new_finite_args, 0.0);
      component_likelihood_functor likelihood_fn;
      result.table().aggregate(table_,
                               make_dim_map(new_finite_args),
                               likelihood_fn);
      return result;
    }

    /**
     * Returns the marginal of this factor over the given set of variables.
     * To ensure that the resulting distribution is not multimodal and is
     * representable by this factor, one of two conditions must hold:
     * 1) no finite variables must be marginalized out from this factor, or
     * 2) all vector variables must be marginalized out from this factor.
     */
    hybrid marginal(const domain& retain) const {
      finite_var_vector new_finite_args = intersect(finite_args(), retain);
      vector_var_vector new_vector_args = intersect(vector_args(), retain);
      if (new_finite_args.size() == num_finite()) { // (1)
        vector_domain retained_vector = make_domain(new_vector_args);
        component_marginal_functor marginal_fn(&retained_vector);
        hybrid result(new_finite_args);
        result.table_.transform(table_, marginal_fn);
        result.args_.insert(retain.begin(), retain.end());
        return result;
      } else if (new_vector_args.empty()) { // (2)
        return hybrid(marginal(make_domain(new_finite_args)));
      } else {
        throw std::invalid_argument("The hybrid marginal may be multimodal. "
                                    "See the documentation on how to avoid this.");
      }
    }

    /**
     * Returns the factor restricted to the given assignment.
     */
    hybrid restrict(const assignment& a) const {
      finite_var_vector new_finite_args = difference(finite_args_, a.finite());
      component_restrict_functor restrict_fn(&a.vector());
      hybrid result(new_finite_args);
      result.table_.restrict(table_,
                             make_index(a.finite(), false /* not strict */),
                             make_dim_map(new_finite_args),
                             restrict_fn);
      const vector_domain& new_vector_args = result.table_.begin()->arguments();
      result.args_.insert(new_vector_args.begin(), new_vector_args.end());
      return result;
    }

    /**
     * Restricts the factor and stores the result to a table factor.
     * The assignment must include all the vector variables in this factor.
     */
    void restrict(const assignment& a, table_factor& result) const {
      finite_var_vector new_finite_args = difference(finite_args_, a.finite());
      component_value_functor value_fn(&a.vector());
      table_factor(new_finite_args).swap(result);
      result.table().restrict(table_,
                              make_index(a.finite(), false /* not strict */),
                              make_dim_map(new_finite_args),
                              value_fn);
    }

    /**
     * Restricts the factor and multiplies the result to a table factor.
     * The assignment must include all the vector variables in this factor.
     */
    void restrict_multiply(const assignment& a, table_factor& result) const {
      table_factor tmp;
      restrict(a, tmp);
      result *= tmp;
    }

    /**
     * Returns a factor with the arguments reordered.
     */
    hybrid reorder(const var_vector& new_order) const {
      finite_var_vector finite_order;
      vector_var_vector vector_order;
      sill::split(new_order, finite_order, vector_order);
      hybrid result(finite_order);
      component_reorder_functor reorder_fn(&vector_order);
      result.table_.join_with(table_, make_dim_map(finite_order), reorder_fn); 
      result.args_.insert(vector_order.begin(), vector_order.end());
      return result;
    }

    /**
     * Returns the normalization constant of this factor.
     */
    result_type norm_constant() const {
      return table_.aggregate(component_likelihood_functor(), real_type(0.0));
    }

    /**
     * Normalizes the factor, so that the likelihoods of all components sum to 0.
     */
    hybrid& normalize() {
      combine_in(norm_constant(), divides_assign<F, result_type>());
      return *this;
    }

    // Learning
    //==========================================================================
    //! Returns the log_likelihood of the dataset given this factor
    real_type log_likelihood(const dataset_type& ds) const {
      factor_evaluator<hybrid> evaluator(*this);
      real_type ll = 0.0;
      foreach (const record_type& r, ds.records(arg_vector())) {
        if (!r.count_missing()) {
          ll += r.weight * log(evaluator(r.values));
        }
      }
      return ll;
    }

    // Private helper functions and functors
    //==========================================================================
  private:
    /**
     * Initializes this factor to given finite and vector argument sequence.
     * Each component is initialized to the given likelihood.
     */
    void initialize(const finite_var_vector& finite_args,
                    const vector_var_vector& vector_args,
                    result_type value) {
      initialize(finite_args, F(vector_args, value));
    }

    /**
     * Initializes this factor to given finite argument sequence.
     * Each component is initialized to the given factor.
     */
    void initialize(const finite_var_vector& finite_args,
                    const F& factor) {
      finite_args_ = finite_args;
      args_ = set_union(make_domain(finite_args_), factor.arguments());

      std::vector<size_t> shape(finite_args.size());
      var_index_.clear();
      for (size_t i = 0; i < finite_args.size(); ++i) {
        var_index_[finite_args[i]] = i;
        shape[i] = finite_args[i]->size();
      }
      dense_table<F>(shape, factor).swap(table_);
    }

    /**
     * Returns the index corresponding to the finite assignment.
     * If strict, each finite argument must be present in the assignment.
     */
    std::vector<size_t> make_index(const finite_assignment& a,
                                   bool strict = true) const {
      std::vector<size_t> result(finite_args_.size(),
                                 std::numeric_limits<size_t>::max());
      for (size_t i = 0; i < finite_args_.size(); ++i) {
        finite_variable* v = finite_args_[i];
        finite_assignment::const_iterator it = a.find(v);
        if (it != a.end()) {
          result[i] = it->second;
        } else if (strict) {
          throw std::invalid_argument(
            "The assignment does not contain the variable " + v->name()
          );
        }
      }
      return result;
    }

    /**
     * Returns the indices of the given finite variables in this factor.
     * All the variables in vars must be in this factor's domain.
     */
    std::vector<size_t> make_dim_map(const finite_var_vector& vars) const {
      std::vector<size_t> result(vars.size());
      for (size_t i = 0; i < vars.size(); ++i) {
        result[i] = safe_get(var_index_, vars[i]);
      }
      return result;
    }

    //! Computes the marginal of the given component over a fixed vector domain
    struct component_marginal_functor {
      const vector_domain* dom;
      component_marginal_functor(const vector_domain* dom) : dom(dom) { }
      F operator()(const F& factor) const { return factor.marginal(*dom); }
    };

    //! Restricts the given component to a fixed vector assignment
    struct component_restrict_functor {
      const vector_assignment* a;
      component_restrict_functor(const vector_assignment* a) : a(a) { }
      F operator()(const F& factor) const { return factor.restrict(*a); }
    };

    //! Evaluates the given component to a fixed value
    struct component_value_functor {
      const vector_assignment* a;
      component_value_functor(const vector_assignment* a) : a(a) { }
      result_type operator()(const F& factor) const { return factor(*a); }
    };

    //! Reorders the arguments of the component
    struct component_reorder_functor {
      const vector_var_vector* v;
      component_reorder_functor(const vector_var_vector* v) : v(v) { }
      F operator()(const F&, const F& f) const { return f.reorder(*v); }
    };

    //! Computes the normalization constant of a given component
    struct component_likelihood_functor {
      result_type operator()(const F& factor) const {
        return factor.norm_constant();
      }
      real_type operator()(real_type value, const F& factor) const {
        return value + factor.norm_constant();
      }
    };

    // Private data members
    //==========================================================================
  private:
    
    //! The type that maps variables to table indices
    typedef std::map<finite_variable*, size_t> var_index_map;

    //! The arguments of this factor (finite and vector)
    domain args_;

    //! The finite arguments of this factor in the internal ordering
    finite_var_vector finite_args_;

    //! The mapping from the finite arguments to the index in finite_args_
    var_index_map var_index_;

    //! The table used to store the components p(X, Y = y)
    dense_table<F> table_;

  }; // class hybrid

  
  // Operator overloads
  //============================================================================

  template <typename F>
  std::ostream& operator<<(std::ostream& out, const hybrid<F>& h) {
    out << h.finite_args() << std::endl;
    out << h.table() << std::endl;
    return out;
  }
  
  template <typename F>
  hybrid<F>& operator*=(hybrid<F>& f, const hybrid<F>& g) {
    if (f.contains(g.finite_args())) {
      f.combine_in(g, multiplies_assign<F>());
    } else {
      hybrid<F>::combine(f, g, std::multiplies<F>()).swap(f);
    }
    return f;
  }

  template <typename F>
  hybrid<F>& operator/=(hybrid<F>& f, const hybrid<F>& g) {
    if (f.contains(g.finite_args())) {
      f.combine_in(g, divides_assign<F>());
    } else {
      hybrid<F>::combine(f, g, std::divides<F>()).swap(f);
    }
    return f;
  }

  template <typename F>
  hybrid<F>& operator*=(hybrid<F>& f, const table_factor& g) {
    if (f.contains(g.arg_vector())) {
      f.combine_in(g, multiplies_assign<F, double>());
    } else {
      hybrid<F>::combine(f, g, sill::multiplies<F, double>()).swap(f);
    }
    return f;
  }

  template <typename F>
  hybrid<F>& operator/=(hybrid<F>& f, const table_factor& g) {
    if (f.contains(g.arg_vector())) {
      f.combine_in(g, divides_assign<F, double>());
    } else {
      hybrid<F>::combine(f, g, sill::divides<F, double>()).swap(f);
    }
    return f;
  }

  template <typename F>
  hybrid<F>& operator*=(hybrid<F>& f, const F& g) {
    f.combine_in(g, multiplies_assign<F>());
    return f;
  }

  template <typename F>
  hybrid<F>& operator/=(hybrid<F>& f, const F& g) {
    f.combine_in(g, divides_assign<F>());
    return f;
  }

  template <typename F>
  hybrid<F>& operator*=(hybrid<F>& f, typename F::result_type v) {
    f.combine_in(v, multiplies_assign<F, typename F::result_type>());
    return f;
  }

  template <typename F>
  hybrid<F>& operator/=(hybrid<F>& f, typename F::result_type v) {
    f.combine_in(v, divides_assign<F, typename F::result_type>());
    return f;
  }
  
  template <typename F>
  hybrid<F> operator*(const hybrid<F>& f, const hybrid<F>& g) {
    return hybrid<F>::combine(f, g, std::multiplies<F>());
  }

  template <typename F>
  hybrid<F> operator/(const hybrid<F>& f, const hybrid<F>& g) {
    return hybrid<F>::combine(f, g, std::divides<F>());
  }

  template <typename F>
  hybrid<F> operator*(const hybrid<F>& h, const table_factor& f) {
    return hybrid<F>::combine(h, f, sill::multiplies<F, double>());
  }

  template <typename F>
  hybrid<F> operator*(const table_factor& f, hybrid<F> h) {
    return hybrid<F>::combine(h, f, sill::multiplies<F, double>());
  }

  template <typename F>
  hybrid<F> operator/(hybrid<F> h, const table_factor& f) {
    return hybrid<F>::combine(h, f, sill::divides<F, double>());
  }

  template <typename F>
  hybrid<F> operator*(const hybrid<F>& h, const F& f) {
    return hybrid<F>::combine(h, f, std::multiplies<F>());
  }

  template <typename F>
  hybrid<F> operator*(const F& f, const hybrid<F>& h) {
    return hybrid<F>::combine(h, f, std::multiplies<F>());
  }

  template <typename F>
  hybrid<F> operator/(const hybrid<F>& h, const F& f) {
    return hybrid<F>::combine(h, f, std::divides<F>());
  }

  template <typename F>
  hybrid<F> operator*(hybrid<F> h, typename F::result_type v) {
    return h *= v;
  }

  template <typename F>
  hybrid<F> operator*(typename F::result_type v, hybrid<F> h) {
    return h *= v;
  }

  template <typename F>
  hybrid<F> operator/(hybrid<F> h, typename F::result_type v) {
    return h /= v;
  }

  // Utility classes
  //============================================================================

  /**
   * Specialization of factor_evaluator for a hybrid factor.
   *
   * The class Instantiates the factor evaluator for each component and
   * delegates the evaluation calls to them.
   */
  template <typename F>
  class factor_evaluator<hybrid<F> > {
  public:
    typedef typename F::result_type  result_type;
    typedef typename F::real_type    real_type;
    typedef hybrid_values<real_type> index_type;
    typedef var_vector               var_vector_type;

    factor_evaluator(const hybrid<F>& factor)
      : offset(factor.table().offset) {
      evaluators.reserve(factor.size());
      foreach (const F& component, factor) {
        evaluators.push_back(factor_evaluator<F>(component));
      }
    }

    result_type operator()(const index_type& index) const {
      size_t i = offset(index.finite);
      return evaluators[i](index.vector);
    }

  private:
    typename dense_table<F>::offset_functor offset;
    std::vector<factor_evaluator<F> > evaluators;

  }; // class factor_evaluator<hybrid<F>>


  /**
   * Specialization of factor_sampler for a hybrid factor.
   *
   * The sampler first computes a table of cumulative sums of the finite
   * component of the distribution. To draw a random sample, the sampler
   * first selects a random component and then delegates to the
   * corresponding component sampler.
   */
  template <typename F>
  class factor_sampler<hybrid<F> > {
  public:
    typedef typename hybrid<F>::index_type index_type;
    typedef var_vector                     var_vector_type;
    typedef typename hybrid<F>::real_type  real_type;

    //! Creates a sampler for a marginal distribution
    explicit factor_sampler(const hybrid<F>& factor) {
      initialize(factor, factor.finite_args(), factor.vector_args());
    }

    //! Creates a sampler for conditional distribution p(head | rest)
    factor_sampler(const hybrid<F>& factor,
                   const var_vector& head) {
      finite_var_vector finite_head;
      vector_var_vector vector_head;
      split(head, finite_head, vector_head);
      initialize(factor, finite_head, vector_head);
    }

    //! Draws a random sample from a marginal distribution
    template <typename RandomNumberGenerator>
    void operator()(index_type& sample, RandomNumberGenerator& rng) const {
      assert(ntail_ == 0);
      cumsum_iterator begin = cumsum_.begin();
      boost::random::uniform_real_distribution<real_type> unif01;
      size_t i = std::upper_bound(begin, begin + nelem_, unif01(rng)) - begin;
      if (i >= nelem_) { i = nelem_ - 1; }
      cumsum_.offset.index(i, nhead_, sample.finite);
      components_[i](sample.vector, rng);
    }

    //! Draws a random sample from a conditional distribution
    template <typename RandomNumberGenerator>
    void operator()(index_type& sample, const index_type& tail,
                    RandomNumberGenerator& rng) const {
      assert(tail.finite.size() == ntail_);
      size_t start = cumsum_.offset(tail.finite, nhead_);
      cumsum_iterator begin = cumsum_.begin() + start;
      boost::random::uniform_real_distribution<real_type> unif01;
      size_t i = std::upper_bound(begin, begin + nelem_, unif01(rng)) - begin;
      if (i >= nelem_) { i = nelem_ - 1; }
      cumsum_.offset.index(i, nhead_, sample.finite);
      components_[i + start](sample.vector, tail.vector, rng);
    }

  private:
    //! Initializes the internal data structures
    void initialize(const hybrid<F>& factor,
                    const finite_var_vector& finite_head,
                    const vector_var_vector& vector_head) {
      // initialize the table and the dimensions
      cumsum_ = dense_table<real_type>(factor.table().shape());
      nhead_ = finite_head.size();
      ntail_ = factor.num_finite() - nhead_;
      nelem_ = num_assignments(finite_head);
      assert(nhead_ <= factor.num_finite());
      assert(std::equal(finite_head.begin(), finite_head.end(),
                        factor.finite_args().begin()));
      assert(cumsum_.size() % nelem_ == 0);

      // compute cumulative sums of factor likelihoods
      typename dense_table<F>::const_iterator src_it = factor.begin();
      typename dense_table<real_type>::iterator dest_it = cumsum_.begin();
      while (dest_it != cumsum_.end()) {
        real_type sum = 0.0;
        for (size_t i = 0; i < nelem_; ++i) {
          sum += src_it->norm_constant();
          *dest_it = sum;
          ++src_it;
          ++dest_it;
        }
        assert(std::abs(sum - 1.0) < 1e-10); // is this stable?
      }

      // initialize the component samplers
      components_.reserve(factor.size());
      foreach (const F& f, factor.table()) {
        components_.push_back(factor_sampler<F>(f, vector_head));
      }
    }
    
    typedef typename dense_table<real_type>::const_iterator cumsum_iterator;
    dense_table<real_type> cumsum_;
    size_t nhead_;
    size_t ntail_;
    size_t nelem_;
    std::vector<factor_sampler<F> > components_;

  }; // class factor_sampler<hybrid<F>>

  /**
   * Specialization of factor_mle_incremental for a hybrid factor.
   * The component factor must support incremental MLE.
   */
  template <typename F>
  class factor_mle_incremental<hybrid<F> > {
  public:
    typedef typename F::real_type    real_type;
    typedef var_vector               var_vector_type;
    typedef hybrid_values<real_type> index_type;

    struct param_type {
      typedef typename factor_mle_incremental<F>::param_type comp_param_type;
      real_type smoothing;
      comp_param_type comp_params;
      param_type(real_type smoothing = 0.0,
                 const comp_param_type& params = comp_param_type())
        : smoothing(smoothing), comp_params(params) { }
    };

    factor_mle_incremental(const var_vector& args,
                           const param_type& params = param_type()) {
      finite_var_vector finite_args;
      vector_var_vector vector_args;
      split(args, finite_args, vector_args);
      hybrid<F>(finite_args).swap(factor_);
      factor_mle_incremental<F> mle(vector_args, params.comp_params);
      size_t n = vector_size(vector_args);
      mle.process(arma::zeros(n), params.smoothing); // fix types
      estimators_.assign(factor_.size(), mle);
    }

    // TODO: note about the form this learns
    factor_mle_incremental(const var_vector& head,
                           const var_vector& tail,
                           const param_type& params = param_type()) {
      finite_var_vector finite_head, finite_tail;
      vector_var_vector vector_head, vector_tail;
      split(head, finite_head, vector_head);
      split(tail, finite_tail, vector_tail);
      hybrid<F>(concat(finite_head, finite_tail)).swap(factor_);
      factor_mle_incremental<F> mle(vector_head, vector_tail, params.comp_params);
      size_t n = vector_size(vector_head) + vector_size(vector_tail);
      mle.process(arma::zeros(n), params.smoothing); // fix types
      estimators_.assign(factor_.size(), mle);
      finite_tail_ = finite_tail;
    }

    factor_mle_incremental(const var_vector& head,
                           const finite_var_vector& tail,
                           const param_type& params = param_type()) {
      finite_var_vector finite_head;
      vector_var_vector vector_head;
      split(head, finite_head, vector_head);
      hybrid<F>(concat(finite_head, tail)).swap(factor_);
      factor_mle_incremental<F> mle(vector_head, params.comp_params);
      size_t n = vector_size(vector_head);
      mle.process(arma::zeros(n), params.smoothing); // fix types
      estimators_.assign(factor_.size(), mle);
      finite_tail_ = tail;
    }

    void process(const index_type& values, real_type weight) {
      size_t i = factor_.table().offset(values.finite);
      estimators_[i].process(values.vector, weight);
    }

    void process(const index_type& values, const table_factor& ptail) {
      size_t nhead = values.finite.size();
      size_t ntail = finite_tail_.size();
      assert(ptail.arg_vector() == finite_tail_);
      assert(nhead + ntail == factor_.num_finite());
      size_t i = factor_.table().offset(values.finite, 0);
      size_t increment = factor_.table().offset.get_multiplier(nhead);
      foreach(double w, ptail.table()) {
        estimators_[i].process(values.vector, w);
        i += increment;
      }
    }

    hybrid<F>& estimate() {
      for (size_t i = 0; i < factor_.size(); ++i) {
        factor_[i] = estimators_[i].estimate();
        factor_[i] *= estimators_[i].weight();
      }
      factor_.update_arguments();
      if (finite_tail_.empty()) {
        return factor_.normalize();
      } else {
        return factor_ /= factor_.marginal(make_domain(finite_tail_));
      }
    }

    real_type weight() const {
      real_type sum = 0.0;
      for (size_t i = 0; i < estimators_.size(); ++i) {
        sum += estimators_[i].weight();
      }
      return sum;
    }

  private:
    hybrid<F> factor_;
    finite_var_vector finite_tail_;
    std::vector<factor_mle_incremental<F> > estimators_;

  }; // class factor_mle_incremental

  // Traits
  //============================================================================

  //! \addtogroup factor_traits
  //! @{
  
  template <typename F>
  struct has_multiplies<hybrid<F> > : public has_multiplies<F> { };

  template <typename F>
  struct has_multiplies_assign<hybrid<F> > : public has_multiplies_assign<F> { };

  template <typename F>
  struct has_divides<hybrid<F> > : public has_divides<F> { };

  template <typename F>
  struct has_divides_assign<hybrid<F> > : public has_divides_assign<F> { };

  template <typename F>
  struct has_arg_max<hybrid<F> > : public has_arg_max<F> { };
  
  template <typename F>
  struct has_marginal<hybrid<F> > : public boost::true_type { };

  //! @}

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

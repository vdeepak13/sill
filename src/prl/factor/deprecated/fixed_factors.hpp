#ifndef PRL_FIXED_FACTORS_HPP
#define PRL_FIXED_FACTORS_HPP

#include <iosfwd>

#include <boost/array.hpp>

#include <prl/global.hpp>
#include <prl/factor/factor.hpp>
#include <prl/factor/constant_factor.hpp>
#include <prl/factor/table_factor.hpp>

#include <prl/range/algorithm.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  template <std::size_t N>
  class binary_factor; // Forward declaration

  /**
   * A discrete factor over a single argument with a fixed number of values.
   * To get the maximum performance, this class should be compiled with -O3.
   *
   * \ingroup factor_types
   * @see Factor
   */
  template <std::size_t N>
  class unary_factor : public factor {

    // Public type declarations
    //==========================================================================
  public:
    //! implements Factor::result_type
    typedef double result_type;

    //! implements Factor::domain_type
    typedef finite_domain domain_type;

    //! implements Factor::variable_type
    typedef finite_variable variable_type;

    //! implements Factor::collapse_type
    typedef constant_factor collapse_type;

    //! implements Factor::collapse_ops
    static const unsigned collapse_ops = ~0; // supports all operations

    //! implements Factor::combine_ops
    static const unsigned combine_ops = ~0; // supports all operations

    // Private data members
    //==========================================================================
  private:

    //! The argument
    finite_variable* u;

    //! The values
    boost::array<double, N> val;

    friend class binary_factor<N>;

    // Constructors and conversion operators
    //==========================================================================
  public:
    /**
     * Default constructor.
     * This constructor does not create a valid factor (since the argument
     * is unknown), but it is necessary to satisfy the Factor concept.
     */
    unary_factor() : u() { } // valid()==false // TODO: make private

    //! Creates a unary factor with the given default value
    //! The argument set must have exactly one element
    unary_factor(const finite_domain& args, double default_value = double())
      : u(args.representative()) {
      assert(args.size() == 1 && u->size() == N);
      prl::fill(val, default_value);
    }

    //! Conversion to a table factor
    template <typename Table>
    operator table_factor<Table>() const {
      table_factor<Table> f(arguments(), 0);
      prl::copy(values(), boost::begin(f.values()));
      return f;
    }

    //! Conversion to human-readable representation
    operator std::string() const {
      std::ostringstream out; out << *this; return out.str(); 
    }

    // Accessors
    //==========================================================================
    //! Returns the argument set of this factor 
    finite_domain arguments() const { 
      return finite_domain(u);
    }

    //! Returns the total number of values in this factor
    size_t size() const {
      return N;
    }

    //! Returns true if the factor is initialized
    bool valid() const {
      return u;
    }

    //! Returns the reference to the i-th element
    double& operator[](size_t i) {
      return val[i];
    }

    //! Returns the reference to the i-th element
    const double& operator[](size_t i) const {
      return val[i];
    }

    //! implements Factor::operator()
    double& operator()(const finite_assignment& a) {
      return val[a[u]];
    }

    //! implements Factor::operator()
    double operator()(const finite_assignment& a) const {
      return val[a[u]];
    }

    //! implements DiscreteFactor::operator()
    double& operator()(size_t i, size_t j) {
      assert(false);
      return val[0];
    }

    //! implements DiscreteFactor::operator()
    const double& operator()(size_t i, size_t j) const {
      assert(false);
      return val[0];
    }

    //! implements DiscreteFactor::values()
    boost::array<double, N>& values() {
      return val;
    }

    //! implements DiscreteFactor::values()
    const boost::array<double, N>& values() const {
      return val;
    }

    // Factor operations
    //==========================================================================
    //! Combines a constant factor into this factor with a binary operation
    unary_factor& combine_in(const constant_factor& f, op_type op) {
      assert(valid());
      binary_op<double> combine_op(op);
      for(size_t i = 0; i < N; i++) val[i] = combine_op(val[i], f.value);
      return *this;
    }

    //! Combines the given factor into this factor with a binary operation
    //! The two factors must have the same argument set
    unary_factor& combine_in(const unary_factor& f, op_type op) {
      assert(valid() && u == f.u);
      binary_op<double> combine_op(op);
      for(size_t i=0; i<N; i++) val[i] = combine_op(val[i], f[i]);
      return *this;
    }

    //! Collapses a factor to a subset of its arguments
    //! The resulting argument set must be empty
    constant_factor collapse(const finite_domain& retain, op_type op) const {
      assert(valid() && retain.empty());
      binary_op<double> collapse_op(op);
      double agg = collapse_op.left_identity();
      for(size_t i = 0; i<N; i++) agg = collapse_op(agg, val[i]);
      return agg;
      //return prl::accumulate(val, op.left_identity(), op);
    }

    //! implements Factor::restrict
    //! The assignment must include a value for this factor's argument
    constant_factor restrict(const finite_assignment& a) const {
      return constant_factor(a[u]);
    }

    //! implements Factor::subst_args
    unary_factor& subst_args(const finite_var_map& map) {
      u = map[u];
      return *this;
    }

    //! Computes a marginal (sum) over a factor expression
    constant_factor marginal(const finite_domain& retain) const {
      return collapse(retain, sum_op);
    }

    //! Returns true if the factor is normalizable
    bool is_normalizable() const {
      return is_positive_finite(norm_constant());
    }

    //! Returns the normalization constant
    double norm_constant() const {
      return marginal(finite_domain::empty_set);
    }

    //! Normalizes a factor in-place.
    unary_factor& normalize() {
      return (*this /= norm_constant());
    }

    //! Returns the maximum value of a factor
    unary_factor maximum(const finite_domain& retain) const {
      return collapse(retain, max_op);
    }

    //! Returns the minimum value of a factor
    unary_factor minimum(const finite_domain& retain) const {
      return collapse(retain, min_op);
    }

  }; // class unary_factor

  // Free functions
  //============================================================================

  //! relates unary_factor
  template <std::size_t N>
  std::ostream& operator<<(std::ostream& out, const unary_factor<N>& f) {
    out << "#F(U|" << f.arguments() << "|";
    prl::copy(f.values(), std::ostream_iterator<double, char>(out, " "));
    out << ")";
    return out;
  }

  template <size_t N>
  struct combine_result< unary_factor<N>, unary_factor<N> > {
    typedef unary_factor<N> type;
  };

  // The default implementation of combine(unary_factor, unary_factor) works

  //! \relates unary_factor
  template <std::size_t N>
  unary_factor<N> weighted_update(const unary_factor<N>& f1,
                                  const unary_factor<N>& f2,
                                  double weight) {
    finite_domain args1 = f1.arguments();
    finite_domain args2 = f2.arguments();
    // Verify that the two factors have the same domain
    assert(args1[0] == args2[0]);
    unary_factor<N> f(args1, 0);
    for(size_t i=0; i<N; i++) f[i] = f1[i] * (1-weight) + f2[i] * weight;
    return f;
  }

  //! Computes the L-1 norm of the difference of two table factors
  //! \relates unary_factor
  template <size_t N>
  double norm_1(const unary_factor<N>& x, const unary_factor<N>& y) {
    assert(false); // not implemented yet
    return 0;
  }

  //! Computes the L-infinity norm of the difference of two table factors
  //! \relates unary_factor
  template <size_t N>
  double norm_inf(const unary_factor<N>& x, const unary_factor<N>& y) {
    assert(false); // not implemented yet
    return 0;
  }

// =================================================================
// =================================================================

  /**
   * A discrete factor over two arguments with a fixed number of values
   * for each variable.
   *
   * To get the maximum performance, this class should be compiled with -O3.
   *
   * \ingroup factor_types
   * \see Factor
   */
  template <std::size_t N = 2>
  class binary_factor : public factor {

    // Public type declarations
    //==========================================================================
  public:
    //! implements Factor::result_type
    typedef double result_type;

    //! implements Factor::domain_type
    typedef finite_domain domain_type;

    //! implements Factor::variable_type
    typedef finite_variable variable_type;

    //! implements Factor::collapse_type
    typedef unary_factor<N> collapse_type;

    //! implements Factor::collapse_ops
    static const unsigned collapse_ops = ~0; // supports all operations

    //! implements Factor::combine_ops
    static const unsigned combine_ops = ~0; // supports all operations
    
    // Shortcut
    typedef unary_factor<N> unary_factor;

    // Private data members
    //==========================================================================
  private:
    //! The argument set of the factor in the natural order (u<v)
    finite_variable *u, *v;

    //! The values of the factor (u,v) in column-major order
    boost::array<double, N*N> val;

    // Constructors and conversion operators
    //==========================================================================
  public:
    /**
     * Default constructor.
     * This constructor does not create a valid factor (since the arguments
     * are unknown), but it is necessary to satisfy the Factor concept.
     */
    binary_factor() : u(), v() { } // valid()==false

    //! Creates a unary factor with the given default value
    //! The argument set must have exactly two variables in it.
    binary_factor(const finite_domain& args, double default_value = double()) {
      assert(args.size() == 2);
      u = args[0];
      v = args[1];
      assert(u->size() == N && v->size() == N);
      prl::fill(val, default_value);
    }

    //! Conversion to a table factor
    template <typename Table>
    operator table_factor<Table>() const {
      table_factor<Table> f(arguments(), 0);
      prl::copy(values(), boost::begin(f.values()));
      return f;
    }

    //! Conversion to human-readable representation
    operator std::string() const {
      std::ostringstream out; out << *this; return out.str(); 
    }

    // Accessors
    //==========================================================================
    //! Returns the argument set of this factor 
    finite_domain arguments() const {
      return make_domain(u, v);
    }

    //! Returns the total number of values in this factor
    size_t size() const {
      return N*N;
    }

    //! Returns true if the factor is initialized
    bool valid() const {
      return u;
    }

    //! implements Factor::operator()
    double& operator()(const finite_assignment& a) {
      return operator()(a[u],a[v]);
    }

    //! implements Factor::operator()
    double operator()(const finite_assignment& a) const {
      return operator()(a[u],a[v]);
    }

    //! implements DiscreteFactor::operator()
    double& operator()(size_t i, size_t j) {
      return val[i+j*N];
    }

    //! implements DiscreteFactor::operator()
    double operator()(size_t i, size_t j) const {
      return val[i+j*N];
    }

    //! implements DiscreteFactor::values()
    boost::array<double, N*N>& values() {
      return val;
    }

    //! implements DiscreteFactor::values()
    const boost::array<double, N*N>& values() const {
      return val;
    }

    // Factor operations
    //==========================================================================
    //! Combines the given factor into this factor with a binary operation
    //! The two factors must have the same argument set
    binary_factor& combine_in(const binary_factor& f, op_type op) {
      assert(valid() && u == f.u && v == f.v);
      binary_op<double> combine_op(op);
      for(size_t i=0; i<val.size(); i++) val[i] = combine_op(val[i], f.val[i]);
      return *this;
    }

    //! Combines this factor with a unary factor.
    //! The argument of f must be present in this factor.
    binary_factor& combine_in(const unary_factor& f, op_type op) {
      assert(valid());
      binary_op<double> combine_op(op);
      if(f.u == u) { // combine along columns
        for(size_t j=0, offset=0; j<N; j++, offset+=N)
          for(size_t i=0; i<N; i++)
            val[offset+i] = combine_op(val[offset+i], f[i]);
      }
      else if(f.u == v) { // combine along rows
        for(size_t i=0; i<N; i++)
          for(size_t j=0, index=i; j<N; j++, index+=N)
            val[index] = combine_op(val[index], f[j]);
      }
      else assert(false); // cannot introduce a new variable
      return *this;
    }

    //! Combines this factor with a constant factor
    //! Since we do not implement a conversion operator from constant_factor,
    //! we need to implement this method here.
    binary_factor& combine_in(const constant_factor& f, 
                              op_type op) {
      assert(valid());
      binary_op<double> combine_op(op);
      for(size_t i=0; i<val.size(); i++) 
        val[i] = combine_op(val[i], f.value);
      return *this; // switch to transform()
    }

    //! Collapses a factor to a single argument
    unary_factor collapse(const finite_domain& retain, op_type op) const {
      assert(valid() && retain.size()==1);
      binary_op<double> collapse_op(op);
      unary_factor f(retain);
      if (retain.representative() == u) { // sum over the rows
        for(size_t i=0; i<N; i++) {
          double result = collapse_op.left_identity();
          for(size_t j=0, index=i; j<N; j++, index+=N)
            result = collapse_op(result, val[index]);
          f[i] = result;
        }
      }
      else if(retain.representative() == v) { // sum over the columns
        const double* start = val.begin();
        for(size_t i = 0; i < N; i++, start += N)
          f[i] = std::accumulate(start, start+N,
                                 collapse_op.left_identity(),
                                 collapse_op);
      }
      else assert(false); // wrong argument
      return f;
    }

    //! implements Factor::restrict
    //! The assignment must restrict exactly one argument.
    //! \todo The function has not been tested
    unary_factor restrict(const finite_assignment& a) const {
      assert(a.contains(u) ^ a.contains(v));  // ^ is xor
      unary_factor f(a.contains(u) ? v : u, 0);
      if(f.u == v) { // restrict to a row
        for(size_t j = 0, index = a[u]; j<N; j++, index += N)
          f[j] = val[index];
      } else { // restrict to a column
        const double* start = val.begin() + size_t(a[v])*N;
        std::copy(start, start+N, f.val.begin());
      }
      return f;
    }

    //! implements Factor::subst_args
    binary_factor& subst_args(const finite_var_map& map) {
      u = map[u];
      v = map[v];
      assert(u < v); // table reorganization not implemented yet
      return *this;
    }

    //! Computes a marginal (sum) over a factor expression
    unary_factor marginal(const finite_domain& retain) const {
      return collapse(retain, sum_op);
    }

    //! Returns true if the factor is normalizable
    bool is_normalizable() const {
      return is_positive_finite(norm_constant());
    }

    //! Returns the normalization constant
    double norm_constant() const {
      return prl::accumulate(val, 0, plus<double>());
    }

    //! Normalizes a factor in-place.
    binary_factor& normalize() { return (*this /= norm_constant()); }

    //! Returns the maximum over rows / columns of this factor
    unary_factor maximum(const finite_domain& retain) const {
      return collapse(retain, max_op);
    }

    //! Returns the minimum over rows / columns of this factor
    unary_factor minimum(const finite_domain& retain) const {
      return collapse(retain, min_op);
    }

  }; // class binary_factor

  // Free functions
  //============================================================================
  
  template <std::size_t N>
  std::ostream& 
  operator<<(std::ostream& out, const binary_factor<N>& f) {
    out << "#F(B|" << f.arguments() << "|";
    prl::copy(f.values(), std::ostream_iterator<double, char>(out, " "));
    out << ")";
    return out;
  }

  // The default implementation of combine for binary_factor works

  template <size_t N>
  struct combine_result< binary_factor<N>, binary_factor<N> > {
    typedef binary_factor<N> type;
  };

  template <size_t N>
  struct combine_result< binary_factor<N>, unary_factor<N> > {
    typedef binary_factor<N> type;
  };

  template <size_t N>
  struct combine_result< unary_factor<N>, binary_factor<N> > {
    typedef binary_factor<N> type;
  };

  //! Combines a binary and a unary factor
  template <size_t N>
  binary_factor<N>
  combine(binary_factor<N> b, const unary_factor<N>& u, op_type op) {
    return b.combine_in(u, op);
  }

  //! Combines a binary and a unary factor
  template <size_t N>
  binary_factor<N>
  combine(const unary_factor<N>& u, binary_factor<N> b, op_type op) {
    return b.combine_in(u, op);
  }

} // namespace prl

#include <prl/macros_undef.hpp>

#endif




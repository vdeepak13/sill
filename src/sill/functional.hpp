#ifndef SILL_FUNCTIONAL_HPP
#define SILL_FUNCTIONAL_HPP

/////////////////////////////////
// This header file is deprecated
/////////////////////////////////

#include <functional>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <iostream>
#include <boost/numeric/conversion/bounds.hpp>

#include <sill/math/operations.hpp>
#include <sill/functional/operators.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! A functor which returns the reciprocal of the given number.
  template <typename T>
  struct reciprocal_functor {

    BOOST_STATIC_ASSERT(std::numeric_limits<T>::has_infinity);

    T operator()(const T& val) const {
      if (val != 0)
        return 1. / val;
      else
        return std::numeric_limits<T>::infinity();
    }

  };

  //! A simple functor that returns a constant value, regardless of the input.
  template <typename T>
  struct constant {
    T value;
  public:
    typedef T result_type;
    explicit constant(T value) : value(value) { }
    T operator[](size_t i) const {
      return value;
    }
    T operator()() const {
      return value;
    }
    template <typename U>
    T operator()(const U&) const {
      return value;
    }
    template <typename U, typename V>
    T operator()(const U&, const V&) const {
      return value;
    }
  };

  //! A simple function for creating constant functors.
  template <typename T>
  constant<T> make_constant(T value) {
    return constant<T>(value);
  }

  // Unary functors
  //============================================================================

#if 0
  //! A simple functor that updates a value using another functor.
  template <typename T, typename Functor>
  struct update : std::unary_function<T, T> {
    Functor f;
    update(Functor f) : f(f) { }
    T& operator()(T& x) const { return (x = f(x)); }
  };

  //! A simple function for creating update functors.
  template <typename Functor>
  update<typename Functor::argument_type, Functor>
  make_update(Functor f) {
    return update<typename Functor::argument_type, Functor>(f);
  }
#endif

  //! A simple functor that returns its input unchanged.
  template <typename T>
  struct identity_t : std::unary_function<T, T> {
    T operator()(const T& value) const { return value; }
  };

  //! A simple functor that returns a pair with second value default_initialized
  template <typename T, typename U>
  struct singleton : std::unary_function<T, std::pair<T, U> > {
    std::pair<T,U> operator()(const T& value) const {
      return make_pair(value, U());
    }
  };

  //! A simple functor that converts its input to a given type
  template <typename T>
  struct converter {
    typedef T result_type;
    template <typename U>
    T operator()(const U& value) const {
      return value;
    }
  };

  //! A unary function that computes \f$-x log(x)\f$.
  template <typename T>
  struct entropy_operator : public std::unary_function<T, T> {
  private:
    double logbase;
  public:
    //! Constructor which uses base e.
    entropy_operator() : logbase(1.0) { }

    //! Constructor which uses the given base.
    explicit entropy_operator(double base) {
      if (base <= 0)
        throw std::logic_error("Log must be defined w.r.t. a positive base.");
      logbase = std::log(base);
    }

    T operator()(T x) {
      if (x > T(0))
        return -x * log(x) / logbase;
      else if (x == T(0))
        return T(0);
      else
        throw std::logic_error("Cannot compute entropy of a negative value.");
    }

    T operator()(T a, T b) {
      return a + operator()(b);
    }
  };

  /**
   * The conjunction operator, which models the symmetric binary
   * operator concept. 
   */
  template <typename T>
  struct logical_and : public std::binary_function<T, T, T> {
    T operator()(const T& a, const T& b) const { return a && b; }
  };


 /** The disjunction operator, which models the symmetric binary
   * operator concept. 
   */
  template <typename T>
  struct logical_or : public std::binary_function<T, T, T> {
    T operator()(const T& a, const T& b) const { return a && b; }
  };

  /**
   * The Kullback-Liebler (KL) divergence operator, \f$f(x, y) = x
   * \log \frac{x}{y}\f$, which models the asymmetric binary operator
   * concept.  It has a left zero element (\f$f(0, a) = 0\f$).
   */
  template <typename T>
  struct kld_operator : public std::binary_function<T, T, T>
  {
    T operator()(const T& a, const T& b) const {
      using std::log;
      
      if (a == T(0))
        return 0;
			double loga= log(a);
      double logb= log(b);
      if (std::isinf(logb)) logb=-700;
      if (std::isinf(loga)) loga=-700;
			double res = a * (loga - logb);
			return res;
    }
  };

  template <typename T>
  struct jsd_operator : public std::binary_function<T, T, T> {
    T operator()(T p, T q) const {
      T m = (p + q) / 2;
      return (kld(p, m) + kld(q, m)) / 2;
    }
  };


  /**
   * The cross entropy operator, \f$f(x, y) = x\log y\f$,
   * It has a left zero element (\f$f(0, a) = 0\f$).
   */
  template <typename T>
  struct cross_entropy_operator : public std::binary_function<T, T, T>
  {
  private:
    double logbase;
  public:
    //! Constructor which uses base e.
    cross_entropy_operator() : logbase(1.) { }

    //! Constructor which uses the given base.
    explicit cross_entropy_operator(double base) {
      if (base <= 0)
        throw std::logic_error("Log must be defined w.r.t. a positive base.");
      logbase = std::log(base);
    }

    T operator()(const T& a, const T& b) const {
      using std::log;
      if (a == T(0))
        return T(0);
      else if (b == T(0))
        return T(std::numeric_limits<double>::infinity());
      else
        return fabs(a * log(b) / logbase);
    }
  };

  /**
   * An operator that computes the absolute difference between two numbers in
   * log space.
   */
  template <typename T>
  struct abs_difference_log : public std::binary_function<T, T, T>
  {
    T operator()(const T& a, const T& b) const {
      using std::log;
      if((double(a)<=std::numeric_limits<double>::min()) || 
         (double(b)<=std::numeric_limits<double>::min())) return 0.0; 
      else return std::abs(log(a) - log(b));
    }
  };


  /**
   * The operator obtained by reversing the arguments of another
   * operator.
   */
  template <typename Op>
  struct reverse_args
    : public std::binary_function< typename Op::second_argument_type,
                                   typename Op::first_argument_type,
                                   typename Op::result_type > {
    typedef typename Op::result_type T;
    static const bool is_symmetric = Op::is_symmetric;
    static const bool has_left_zero = Op::has_right_zero;
    static const bool has_right_zero = Op::has_left_zero;
    static const bool has_left_identity = Op::has_right_identity;
    static const bool has_right_identity = Op::has_left_identity;
    static T left_zero() { return Op::right_zero(); }
    static T right_zero() { return Op::left_zero(); }
    static T left_identity() { return Op::right_identity(); }
    static T right_identity() { return Op::left_identity(); }
    reverse_args() { }
    reverse_args(Op op) : orig_op(op) { }
    T operator()(const T& x, const T& y) const { return orig_op(y, x); }
  private:
    Op orig_op;
  };

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_FUNCTIONAL_HPP

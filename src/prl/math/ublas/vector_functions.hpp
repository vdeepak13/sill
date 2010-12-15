#ifndef PRL_MATH_UBLAS_VECTOR_FUNCTIONS_HPP
#define PRL_MATH_UBLAS_VECTOR_FUNCTIONS_HPP

#include <boost/numeric/ublas/expression_types.hpp>

#include <prl/range/algorithm.hpp>
#include <prl/functional.hpp>
#include <prl/math/ublas/vector_operators.hpp>

#include <prl/range/algorithm.hpp>
#include <prl/range/io.hpp>

#include <prl/macros_def.hpp>

// in namespace boost::numeric::ublas, to take advantage of ADL
namespace boost { namespace numeric { namespace ublas
{

  //! \addtogroup math_ublas
  //! @{

  // Vector aggregates
  //============================================================================
  //! Computes a matrix M-norm of vector V
  //! \todo is this v*M*v or v*M^{-1}*v?
  template <typename VE, typename ME>
  typename VE::value_type
  norm_p(const vector_expression<VE>& v,
         const matrix_expression<ME>& m) {
    //assert(false); //fixme
    return inner_prod(v, prod(m, v));
  }

  //! Computes the square of a norm of a vector
  template <typename E>
  typename E::value_type norm_2s(const vector_expression<E>& v) {
    return inner_prod(v(), v());
  }

  /**
   * Returns the effective rank of a matrix. Here, v is the diagonal
   * obtained by an SVD decomposition.
   * @param precision a multiplicative factor; all numbers max(v)*precision
   *        or less are considered zero.
   */
  template <typename E>
  typename E::size_type
  rank(const vector_expression<E>& v, float precision = 1e-10) {
    static_assert((E::complexity==0));
    typedef typename E::value_type T;
    if (v().empty()) return 0;
    T threshold = max(v()) * precision;
    return prl::count(v() > threshold, true);
  }

  // Element-wise vector operations
  //============================================================================
  //! Applies a unary function to each vector element and returns the result
  //! \todo it is easy to avoid the temporaries here (if needed)
  template <typename E, typename F>
  vector<typename E::value_type>
  transform(const vector_expression<E>& u, F f) {
    vector<typename E::value_type> v(u().size());
    prl::transform(u(), v.begin(), f);
    return v;
  }

  //! Computes the square-root for each element of the vector
  template <typename E>
  vector<typename E::value_type> sqrt(const vector_expression<E>& v) {
    return transform(v, prl::square_root<typename E::value_type>());
  }

  // Vector concatenation
  //============================================================================
  //! Concatenates two vectors
  template <typename E1, typename E2>
  vector<typename E1::value_type> 
  concat(const vector_expression<E1>& e1, 
         const vector_expression<E2>& e2) {
    std::size_t i1 =      e1().size();
    std::size_t i2 = i1 + e2().size();
    vector<typename E1::value_type> v(i2);
    v(range(0,  i1)) = e1();
    v(range(i1, i2)) = e2();
    return v;
  }

  //! Concatenates three vectors
  template <typename E1, typename E2, typename E3>
  vector<typename E1::value_type> 
  concat(const vector_expression<E1>& e1, 
         const vector_expression<E2>& e2,
         const vector_expression<E3>& e3) {
    std::size_t i1 =      e1().size();
    std::size_t i2 = i1 + e2().size();
    std::size_t i3 = i2 + e3().size();
    vector<typename E1::value_type> v(i3);
    v(range(0,  i1)) = e1();
    v(range(i1, i2)) = e2();
    v(range(i2, i3)) = e3();
    return v;
  }

  //! @}

} } } // namespaces

#include <prl/macros_undef.hpp>

#endif

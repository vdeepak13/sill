#ifndef PRL_MATH_UBLAS_VECTOR_OPERATORS_HPP
#define PRL_MATH_UBLAS_VECTOR_OPERATORS_HPP

#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublas/vector.hpp>

#include <prl/range/forward_range.hpp>
#include <prl/range/transformed.hpp>

#include <boost/iterator/iterator_categories.hpp>
#include <functional>
#include <algorithm>

//! Defines operators that compare each element in a vector expression 
//! with a scalar
#define BOOST_UBLAS_VECTOR_SCALAR_COMPARE(oper, functor)		\
  template <typename E>							\
  prl::forward_range<bool>                                              \
  oper(const vector_expression<E>& vec, typename E::value_type val) {	\
    typedef typename E::value_type T;					\
    return make_transformed(vec(), std::bind2nd(std::functor<T>(), val)); \
  }									\
									\
  template <typename E>							\
  prl::forward_range<bool>                                              \
  oper(typename E::value_type val, const vector_expression<E>& vec) {	\
    typedef typename E::value_type T;					\
    return make_transformed(vec(), std::bind2nd(std::functor<T>(), val)); \
  }									

//! Defines a scalar-vector expression
#define BOOST_UBLAS_SCALAR_VECTOR_EXPR(oper, functor)			\
  template<class E2>							\
  BOOST_UBLAS_INLINE							\
  typename vector_binary_scalar1_traits<const typename E2::value_type, E2, functor<typename E2::value_type, typename E2::value_type> >::result_type \
  oper(const typename E2::value_type& e1,				\
       const vector_expression<E2>& e2) {				\
    typedef typename vector_binary_scalar1_traits<const typename E2::value_type, E2, functor<typename E2::value_type, typename E2::value_type> >::expression_type expression_type; \
    return expression_type(e1, e2());					\
  }

//! Defines a vector-scalar expression
#define BOOST_UBLAS_VECTOR_SCALAR_EXPR(oper, functor)			\
  template<class E1>							\
  BOOST_UBLAS_INLINE							\
  typename vector_binary_scalar2_traits<E1, const typename E1::value_type, functor<typename E1::value_type, typename E1::value_type> >::result_type \
  oper(const vector_expression<E1>& e1,					\
       const typename E1::value_type& e2) {				\
    typedef typename vector_binary_scalar2_traits<E1, const typename E1::value_type, functor<typename E1::value_type, typename E1::value_type> >::expression_type expression_type; \
    return expression_type (e1(), e2);					\
  }

namespace boost { namespace numeric { namespace ublas
{

  // Vector-scalar comparisons
  BOOST_UBLAS_VECTOR_SCALAR_COMPARE(operator<, less);
  BOOST_UBLAS_VECTOR_SCALAR_COMPARE(operator<=, less_equal);
  BOOST_UBLAS_VECTOR_SCALAR_COMPARE(operator>, greater);
  BOOST_UBLAS_VECTOR_SCALAR_COMPARE(operator>=, greater_equal);
  BOOST_UBLAS_VECTOR_SCALAR_COMPARE(operator==, equal_to);
  BOOST_UBLAS_VECTOR_SCALAR_COMPARE(operator!=, not_equal_to);
  
  // A few obvious vector-scalar expressions that are missing from uBLAS
  BOOST_UBLAS_SCALAR_VECTOR_EXPR(operator+, scalar_plus);
  BOOST_UBLAS_SCALAR_VECTOR_EXPR(operator-, scalar_minus);
  BOOST_UBLAS_SCALAR_VECTOR_EXPR(operator/, scalar_divides);
  BOOST_UBLAS_VECTOR_SCALAR_EXPR(operator+, scalar_plus);
  BOOST_UBLAS_VECTOR_SCALAR_EXPR(operator-, scalar_minus);

  // Compares two vector expressions
  //! \ingroup math_ublas
  template <typename T>
  bool operator==(const vector<T>& v1, const vector<T>& v2) {
    return v1.size() == v2.size() 
      && std::equal(v1.begin(), v1.end(), v2.begin());
  }
  
  
} } }

#undef BOOST_UBLAS_VECTOR_COMPARE

#endif

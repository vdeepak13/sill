#ifndef PRL_MATH_UBLAS_MATRIX_FUNCTIONS_HPP
#define PRL_MATH_UBLAS_MATRIX_FUNCTIONS_HPP

#include <algorithm>

#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/banded.hpp>

#include <prl/range/algorithm.hpp>
#include <prl/range/numeric.hpp>

#include <prl/macros_def.hpp>

namespace boost { namespace numeric { namespace ublas 
{

  //! \addtogroup math_ublas
  //! @{

  // Matrix properties
  //============================================================================
  // Returns the number of elements of a matrix
  template <typename E>
  typename E::size_type numel(const matrix_expression<E>& m) {
    return m().size1()*m().size2();
  }

  // Returns true if the matrix has zero elements.
  template <typename E>
  bool isempty(const matrix_expression<E>& m) {
    return numel(m)==0;
  }

  //! Returns true if the expression represents a square matrix
  template <typename E>
  bool issquare(const matrix_expression<E>& m) {
    return m().size1() == m().size2();
  }

  // Matrix aggregates
  //============================================================================
  /**
   * Returns the aggregate of a matrix along the given dimension.
   * @param dim 0 (returns a row vector) or 1 (returns a column vector)
   * @param f a binary function
   */
  template <typename M, typename F>
  vector<typename M::value_type>
  aggregate(const matrix_expression<M>& e, 
            typename M::value_type zero, 
            F f, int dim) {
    typedef typename M::value_type T;
    concept_assert((BinaryFunction<F,T,T,T>));
    assert(dim == 1 || dim == 2);

    const M& m = e();
    vector<T> v(dim == 1 ? m.size2() : m.size1());

    if (dim == 1) { // aggregate along each column
      for(std::size_t i = 0; i < m.size2(); i++)
        v[i] = prl::accumulate(m.column(i), zero, f);
    } 
    else { // aggregate along each row
      for(std::size_t i = 0; i < m.size1(); i++)
        v[i] = prl::accumulate(m.row(i), zero, f);
    }
    return v;
  }
  
  //! Returns the sum of a matrix along the given dimension
  template <typename M>
  vector<typename M::value_type> sum(const matrix_expression<M>& e, int dim) {
    return aggregate(e, 0, std::plus<typename M::value_type>(), dim);
  }

  //! Returns the product of a matrix along the given dimension
  template <typename M>
  vector<typename M::value_type> prod(const matrix_expression<M>& e, int dim) {
    return aggregate(e, 1, std::multiplies<typename M::value_type>(), dim);
  }

  // Diagonal matrices
  //============================================================================
  //! Returns the diagonal of a matrix
  template <typename E>
  vector<typename E::value_type> diag(const matrix_expression<E>& m) {
    vector<typename E::value_type> v(std::min(m().size1(), m().size2()));
    for(std::size_t i = 0; i<v.size(); i++) v[i] = m()(i,i);
    return v;
  }

  //! Returns the diagonal matrix for a given vector expression
  template <typename E>
  diagonal_matrix<typename E::value_type>
  diag(const vector_expression<E>& v) {
    diagonal_matrix<typename E::value_type> d(v().size());
    prl::copy(v(), d.data().begin());
    return d;
  }

  // Matrix replication and concatenation
  //============================================================================
  //! Concatenates two matrices horizontally
  template <typename E1, typename E2>
  matrix<typename E1::value_type, column_major> 
  horzcat(const matrix_expression<E1>& e1,
          const matrix_expression<E2>& e2) {
    assert(e1().size1() == e2().size1());
    std::size_t c1 =      e1().size2();
    std::size_t c2 = c1 + e2().size2();
    matrix<typename E1::value_type, column_major> a(e1().size1(), c2);
    a(range::all(), range(0,  c1)) = e1();
    a(range::all(), range(c1, c2)) = e2();
    return a;
  }

  //! Concatenates three matrices horizontally
  template <typename E1, typename E2, typename E3>
  matrix<typename E1::value_type, column_major> 
  horzcat(const matrix_expression<E1>& e1,
          const matrix_expression<E2>& e2,
          const matrix_expression<E3>& e3) {
    assert(e1().size1() == e2().size1() &&
           e1().size1() == e3().size1());
    std::size_t c1 =      e1().size2();
    std::size_t c2 = c1 + e2().size2();
    std::size_t c3 = c2 + e3().size2();
    matrix<typename E1::value_type, column_major> a(e1().size1(), c3);
    a(range::all(), range(0,  c1)) = e1();
    a(range::all(), range(c1, c2)) = e2();
    a(range::all(), range(c2, c3)) = e3();
    return a;
  }

  //! Concatenates two matrices vertically
  template <typename E1, typename E2>
  matrix<typename E1::value_type, column_major> 
  vertcat(const matrix_expression<E1>& e1,
          const matrix_expression<E2>& e2) {
    assert(e1().size2() == e2().size2());
    std::size_t r1 =      e1().size1();
    std::size_t r2 = r1 + e2().size1();
    matrix<typename E1::value_type, column_major> a(r2, e1().size2());
    a(range(0,  r1), range::all()) = e1();
    a(range(r1, r2), range::all()) = e2();
    return a;
  }

  //! Concatenates three matrices vertically
  template <typename E1, typename E2, typename E3>
  matrix<typename E1::value_type, column_major> 
  vertcat(const matrix_expression<E1>& e1,
          const matrix_expression<E2>& e2,
          const matrix_expression<E3>& e3) {
    assert(e1().size2() == e2().size2());
    std::size_t r1 =      e1().size1();
    std::size_t r2 = r1 + e2().size1();
    std::size_t r3 = r2 + e3().size1();
    matrix<typename E1::value_type, column_major> a(r3, e1().size2());
    a(range(0,  r1), range::all()) = e1();
    a(range(r1, r2), range::all()) = e2();
    a(range(r2, r3), range::all()) = e3();
    return a;
  }

  /**
   * Replicates a vector m by n times
   * \param transpose if false (default), replicates a column vector, 
   *                  if true, replicates a row vector.
   */
  template <typename E>
  matrix<typename E::value_type, column_major>
  repmat(const vector_expression<E>& e,
         std::size_t m,
         std::size_t n,
         bool transpose = false) {
    std::size_t len = e().size();
    matrix<typename E::value_type, column_major> a;
    if (!transpose) { // column vector
      a.resize(m*len, n);
      vector<typename E::value_type> v(m*len);
      std::size_t k = 0;
      for(std::size_t i = 0; i < m; i++, k += len)
        v(range(k, k + len)) = e();
      for(std::size_t j = 0; j < n; j++) 
        a.column(j) = v;
    } else { // row vector
      a.resize(m, n*len);
      vector<typename E::value_type> v(n*len);
      std::size_t k = 0;
      for(std::size_t j = 0; j < n; j++, k += len)
        v(range(k, k + len)) = e();
      for(std::size_t i = 0; i < m; i++) 
        a.row(i) = v;
    }
    return a;
  }

  //! }@
    
} } } // namespaces

#include <prl/macros_undef.hpp>

#endif

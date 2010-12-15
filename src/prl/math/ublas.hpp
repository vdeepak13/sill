#ifndef PRL_MATH_UBLAS_HPP
#define PRL_MATH_UBLAS_HPP

// Standard types and functions from uBLAS
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/detail/concepts.hpp> // for vector addition

// Additional functions
#include <prl/math/ublas/serialize_storage.hpp>
#include <prl/math/ublas/matrix_functions.hpp>
#include <prl/math/ublas/vector_functions.hpp>
#include <prl/math/ublas/io.hpp>

// Lapack bindings using Boost Sandbox
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/bindings/traits/ublas_vector.hpp>
#include <prl/math/ublas/determinant.hpp>
#include <prl/math/ublas/inverse.hpp>
#include <prl/math/ublas/eig.hpp>
#include <prl/math/ublas/svd.hpp>
#include <prl/math/ublas/chol.hpp>
#include <prl/math/ublas/solve.hpp>

#include <prl/serializable.hpp>
// #include <prl/range/numeric.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  //! \ingroup math_ublas
  namespace ublas = boost::numeric::ublas;

  /**
   * A class that implements the LinearAlgebra concept using Boost.uBLAS.
   * \ingroup math_ublas
  */
  template <typename T>
  class ublas_algebra {
  public:
    //! implements LinearAlgebra::value_type
    typedef T value_type;

    //! implements LinearAlgebra::matrix_type
    typedef ublas::matrix<T, ublas::column_major> matrix_type;

    //! implements LinearAlgebra::vector_type
    typedef ublas::vector<T> vector_type;

    //! implements LinearAlgebra::index_range
    typedef ublas::range index_range;
    
    //! implements LinearAlgebra::index_array
    typedef ublas::indirect_array< std::vector<size_t> > index_array;

    //! implements LinearAlgebra::vector_reference
    typedef ublas::vector_indirect<vector_type, index_array> vector_reference;

    //! implements LineraAlgebra::matrix_reference
    typedef ublas::matrix_indirect<matrix_type, index_array> matrix_reference;

    //! implements LinearAlgebra::vector_const_reference
    typedef const ublas::vector_indirect<const vector_type, index_array>
      vector_const_reference;

    //! implements LineraAlgebra::matrix_const_reference
    typedef const ublas::matrix_indirect<const matrix_type, index_array> 
      matrix_const_reference;

    //! implements LinearAlgebra::range
    static ublas::range range(size_t start, size_t stop) {
      return ublas::range(start, stop);
    }

    //! implements LinearAlgebra::identity
    static ublas::identity_matrix<T> identity(size_t size) {
      return ublas::identity_matrix<T>(size);
    }
    
    //! implements LinearAlgebra::unit_vector
    static ublas::unit_vector<T> unit_vector(size_t size, size_t index) {
      return ublas::unit_vector<T>(size, index);
    }

    //! implements LinearAlgebra::zeros
    static ublas::zero_vector<T> zeros(size_t size) {
      return ublas::zero_vector<T>(size);
    }

    //! implements LinearAlgebra::zeros
    static ublas::zero_matrix<T> zeros(size_t size1, size_t size2) {
      return ublas::zero_matrix<T>(size1, size2);
    }

    //! implements LinearAlgebra::ones
    static ublas::scalar_vector<T> ones(size_t size) {
      return ublas::scalar_vector<T>(size, 1);
    }

    //! implements LinearAlgebra::ones
    static ublas::scalar_matrix<T> ones(size_t size1, size_t size2) {
      return ublas::scalar_matrix<T>(size1, size2, 1);
    }

    //! implements LinearAlgebra::scalars
    static ublas::scalar_vector<T> scalars(size_t size, T value) {
      return ublas::scalar_vector<T>(size, value);
    }

    //! implements LinearAlgebra::scalars
    static ublas::scalar_matrix<T> scalars(size_t size1, size_t size2, T value){
      return ublas::scalar_matrix<T>(size1, size2, value);
    }

    //! implements LinearAlgebra::linspace
    static vector_type linspace(value_type start, value_type stop, size_t n) {
      assert(start <= stop);
      if (n == 0) return vector_type();
      if (n == 1) return vector_type(1, stop);
      vector_type v(n);
      double delta = (stop-start) / (n-1);
      for(size_t i = 0; i < n; i++)
        v[i] = start + i*delta;
      return v;
    }

  }; // class ublas_algebra

  //! A linear algebra kernel with double number representation
  //! \ingroup math_ublas
  typedef ublas_algebra<double> ublas_double;

  //! A linear algebra kernel with float number representation
  //! \ingroup math_ublas
  typedef ublas_algebra<float> ublas_float;

  //EXPORT_PRIMITIVE2(prl::ublas_double, "ublas_double");
  //EXPORT_PRIMITIVE2(prl::ublas_float, "ublas_float");

  
  //! Concatenates a sequence of vectors
  template <typename R>
  typename R::value_type concat(const R& vectors) {
    typedef typename boost::remove_const<typename R::value_type>::type vector;

    // compute the size of the resulting vector
    size_t n = 0;
    foreach(const vector& v, vectors) n += v.size();
    vector result(n);

    // assign the vectors to the right indices
    n = 0;
    foreach(const vector& v, vectors) {
      subrange(result, n, n+v.size()) = v;
      n += v.size();
    }
    return result;
  }

} // namespace prl

#include <prl/macros_undef.hpp>

#endif
  

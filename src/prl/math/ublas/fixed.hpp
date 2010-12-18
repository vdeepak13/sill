#ifndef SILL_MATH_UBLAS_FIXED_HPP
#define SILL_MATH_UBLAS_FIXED_HPP

#include <algorithm>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

#include <sill/macros_def.hpp>

namespace boost { namespace numeric { namespace ublas
{
  //! A fixed-length vector
  //! \ingroup math_ublas
  template<typename T, std::size_t N>
  class fixed_vector : public bounded_vector<T, N> {
  public:
    //! The base type
    typedef bounded_vector<T, N> base;
    using base::data;

    typedef typename base::size_type size_type;
    typedef typename base::const_reference const_reference;

  private:
    // prevent accidental reshaping
    void resize(size_type size1, size_type size2, bool preserve);
    void insert_element(size_type i, size_type j, const_reference t);
    void erase_element(size_type i, size_type j);

  public:
    //! Default constructor; the elements are not initialized
    fixed_vector() : base(N) { }

    //! Constant-initialization
    fixed_vector(const T& value) : base(N) {
      std::fill_n(data().begin(), N, value);
    }

    //! constructs a 3-vector
    //! Is there any way to write this in a general & safe way?
    fixed_vector(T a, T b, T c) : base(N) {
      static_assert(N==3);
      (*this)(0) = a;
      (*this)(1) = b;
      (*this)(2) = c;
    }
      
    // Conversion from a vector expression
    template <typename E>
    fixed_vector(const vector_expression<E>& v) : base(v) {
      assert(v().size() == N);
    }
    
    // Assignment from a vector expression
    template <typename E>
    fixed_vector& operator=(const vector_expression<E>& v) {
      base::operator=(v());
      return *this;
    }

  }; // class fixed_vector

  //! A fixed-size matrix
  //! \ingroup math_ublas
  template<typename T, std::size_t M, std::size_t N = M, typename D = row_major>
  class fixed_matrix : public bounded_matrix<T, M, N, D> {
  public:
    //! The base type
    typedef bounded_matrix<T, M, N, D> base;

    typedef typename base::size_type size_type;
    typedef typename base::const_reference const_reference;

    // prevent accidental reshaping
    void resize(size_type size, bool preserve);
    void insert_element(size_type i, const_reference t);
    void erase_element (size_type i);

  public:
    //! Default constructor; does not initialize the elements
    fixed_matrix() : base(M,N) { }

    //! Constant-initialization
    fixed_matrix(const T& value) : base(M,N) {
      std::fill_n(this->data().begin(), M*N, value);
    } 

    //! Conversion from a matrix expression
    template <typename E>
    fixed_matrix(const matrix_expression<E>& m) : base(m) {
      assert(m().size1() == M);
      assert(m().size2() == N);
    }
    
    //! Assignment from a matrix expression
    template <typename E>
    fixed_matrix& operator=(const matrix_expression<E>& m) {
      base::operator=(m()); // performs bounds checking
      return *this;
    }
  }; // class fixed_matrix


} } } // namespaces

#include <sill/macros_undef.hpp>

#endif

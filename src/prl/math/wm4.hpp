#ifndef SILL_MATH_BINDINGS_WM4_HPP
#define SILL_MATH_BINDINGS_WM4_HPP

#include <boost/numeric/ublas/fixed_container.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_functions.hpp>
#include <boost/numeric/ublas/vector_functions.hpp>

#include <algorithm>
#include <iosfwd>

#include <sill/detail/tuple.hpp>
#include <sill/math/bindings/lapack.hpp> // fill in for missing functions

#include <Wm4Vector2.h>
#include <Wm4Vector3.h>
#include <Wm4Matrix2.h>
#include <Wm4Matrix3.h>
#include <Wm4Quaternion.h>

#include <sill/macros_def.hpp>

// This file may not compile at the moment
// TODO: kill the namespace soup

// WildMagic4 interface 

namespace sill { namespace math { namespace bindings
{
  namespace wm4
  {
    namespace ublas = boost::numeric::ublas;
    // using boost::enable_if_c;
    
    // forward declaration
    template <typename T, std::size_t N> class vector;
    
    namespace impl {
      // Vector type deduction
      template <typename T, std::size_t N> struct vec { };
      template <typename T> struct vec<T,2> {typedef Wm4::Vector2<T> type; };
      template <typename T> struct vec<T,3> {typedef Wm4::Vector3<T> type; };
      
      // Matrix type deduction
      template <typename T, std::size_t N> struct mat { };
      template <typename T> struct mat<T,2> {typedef Wm4::Matrix2<T> type; };
      template <typename T> struct mat<T,3> {typedef Wm4::Matrix3<T> type; };

      // Row-major bool deduction
      template <typename D> struct row_major { };
      template<> struct row_major<ublas::row_major> {
	static const bool value = true; };
      template<> struct row_major<ublas::column_major> {
	static const bool value = false;};
      
      // Extracts the diagonal of a WM4 matrix
      template<typename T>
      vector<T,2> diag(const Wm4::Matrix2<T>& m) { 
	return vector<T,2>(m(0,0), m(1,1));
      }
      template <typename T>
      vector<T,3> diag(const Wm4::Matrix3<T>& m) {
	return vector<T,3>(m(0,0), m(1,1), m(2,2));
      }
    };

    //! WM4-compatible vector class
    template <typename T, std::size_t N>
    class vector : public ublas::fixed_vector<T,N> {
    public:
      //! The corresponding Wm4 vector type
      typedef typename impl::vec<T,N>::type wm4_vector;

      //! The base type
      typedef ublas::fixed_vector<T,N> base;
      using base::data;
      typedef typename base::size_type size_type;

      //! Default constructor; does not initialize the elements.
      vector() { };

      //! Constructor that takes in the size; the size must match N
      vector(size_type size) { assert(size == N); }

      //! Constant constructor
      vector(size_type size, T value) : base(value) { assert(size == N); }
    
      //! Initialization of 2D vectors
      vector(T x, T y) {
	static_assert((N==2));
	(*this)[0] = x; (*this)[1] = y;
      }

      //! Initialization of 3D vectors
      vector(T x, T y, T z) {
	static_assert((N==3));
	(*this)[0] = x; (*this)[1] = y; (*this)[2] = z;
      }

      //! Conversion from a vector expression
      template <typename E>
      vector(const ublas::vector_expression<E>& v) : base(v) { }

      //! Conversion from a WM4 vector object
      vector(const wm4_vector& v) { 
	const T* v_begin = v;
	std::copy(v_begin, v_begin+N, data().begin());
      }

      //! Conversion to a WM4 vector object
      operator wm4_vector() const { 
	return wm4_vector(data().begin()); 
      }

      //! Conversion to a WM4 vector object
      wm4_vector to_wm4() const { return *this; }
    };

    //! WM4-compatible matrix class
    template <typename T, std::size_t N, typename D = ublas::row_major>
    class matrix : public ublas::fixed_matrix<T,N,N,D> {
    public:
      //! The corresponding WM4 matrix type
      typedef typename impl::mat<T,N>::type wm4_matrix;

      //! The base type
      typedef ublas::fixed_matrix<T,N,N,D> base;
      using base::data;
      typedef typename base::size_type size_type;

      //! Default constructor; does not initialize the elements.
      matrix() { };

      //! Standard matrix constructor; the dimensions must match N
      matrix(size_type m, size_type n) { assert(m == N && n == N); }

      //! Initialization of a rotation matrix from a quaternion
      matrix(const Wm4::Quaternion<T>& q) {
        wm4_matrix m;
        q.ToRotationMatrix(m);
        (*this) = m;
      }

      //! Conversion from a matrix expression
      template <typename E>
      matrix(const ublas::matrix_expression<E>& m) : base(m) { }
      
      //! Conversion from a WM4 matrix object
      matrix(const wm4_matrix& m) { 
	if (impl::row_major<D>::value) { // WM4 stores data in row-major order
	  const T* m_begin = m;
	  std::copy(m_begin, m_begin + N*N, data().begin());
	} else {
	  wm4_matrix mt = m.Transpose();
	  const T* mt_begin = mt;
	  std::copy(mt_begin, mt_begin + N*N, data().begin());
	}
      }

      //! Conversion to a WM4 matrix object
      operator wm4_matrix() const {
	return wm4_matrix(data().begin(), impl::row_major<D>::value);
      }

      //! Conversion to a WM4 matrix object
      wm4_matrix to_wm4() const { return *this; }
    };

    //! Returns the inverse of a fixed-size matrix
    template <typename T, std::size_t N, typename D>
    matrix<T,N,D> inv(const matrix<T,N,D>& m) {
      return m.to_wm4().Inverse();
    }
  
    //! Returns the determinant of a fixed-size matrix
    template <typename T, std::size_t N, typename D>
    T det(const matrix<T,N,D>& m) {
      return m.to_wm4().Determinant();
    }

    //! Returns the adjoint of a fixed-size matrix
    template <typename T, std::size_t N, typename D>
    matrix<T,N,D> adj(const matrix<T,N,D>& m) {
      return m.to_wm4().Adjoint();
    }

    /*
    //! Returns a SVD (U,S,V) of a 3x3 matrix
    //! \deprecated wm4's implementation appears to be unstable
    template <typename T, typename D>
    tuple<matrix<T,3,D>, vector<T,3>, matrix<T,3,D> >
    svd(const matrix<T,3,D>& m) {
      Wm4::Matrix3<T> u,s,vt;
      m.to_wm4().SingularValueDecomposition(u, s, vt);
      return make_tuple(u, impl::diag(s), vt.Transpose());
    }*/

    //! Returns the eigenvalue decomposition (R,D) of a PSD matrix
    template <typename T, std::size_t N, typename D>
    pair<matrix<T,N,D>, vector<T,N> > eig(const matrix<T,N,D>& m) {
      typename matrix<T,N,D>::wm4_matrix r,d;
      make_wm4(m).EigenDecomposition(r, d);
      return std::make_pair(r, impl::diag(d));
    }
  
  } // namespace wm4

  namespace ublas = boost::numeric::ublas;
  
  template <typename T, std::size_t N>
  class wm4_kernel {
  public:
    typedef T value_type;
    typedef std::size_t size_type;
    typedef wm4::matrix<T,N,ublas::column_major> matrix;
    typedef wm4::matrix<T,N,ublas::column_major> symmetric_matrix;
    typedef wm4::vector<T,N> vector;
    
    static ublas::identity_matrix<T> identity(size_type size) {
      assert(size==N);
      return ublas::identity_matrix<T>(N);
    }
    
    static ublas::unit_vector<T> unit_vector(size_type size, size_type index) {
      assert(size==N);
      return ublas::unit_vector<T>(N, index);
    }

    // etc.
      
  };

} } } // namespaces

namespace Wm4 {

  template <typename Char, typename Traits, typename T>
  std::basic_ostream<Char,Traits>&
  operator<<(std::basic_ostream<Char,Traits>& out, const Quaternion<T>& q) {
    out << "[" << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << "]";
    return out;
  }

}

#include <sill/macros_undef.hpp>

#endif

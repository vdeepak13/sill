#ifndef SILL_MATH_UBLAS_QUATERNION_HPP
#define SILL_MATH_UBLAS_QUATERNION_HPP

#include <iterator>
#include <cmath>

#include <boost/math/quaternion.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublas/detail/iterator.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <sill/global.hpp>
#include <sill/range/numeric.hpp>
#include <sill/math/ublas/fixed.hpp>

#include <sill/range/algorithm.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  namespace ublas = boost::numeric::ublas;
  
  //! A quaternion class that also models a Vector Expression concept
  //! \ingroup math_ublas
  template <typename T>
  class quaternion
    : public boost::math::quaternion<T>,
      public ublas::vector_expression< quaternion<T> > {
  private:
    typedef quaternion self_type;
    typedef ublas::fixed_vector<T,3> vector3;

  public:
    typedef boost::math::quaternion<T> base_type;
    typedef T value_type;
    typedef T& reference;
    typedef const T& const_reference;
    typedef unsigned size_type;
    typedef int difference_type;
    typedef ublas::indexed_const_iterator<
      quaternion,
      ublas::dense_random_access_iterator_tag> const_iterator;
    typedef ublas::indexed_iterator<
      quaternion,
      ublas::dense_random_access_iterator_tag> iterator;
    typedef const ublas::vector_reference<const self_type> const_closure_type;
    typedef ublas::vector_reference<self_type> closure_type;
    typedef ublas::dense_tag storage_category;

    //! Standard constructor
    //! in the future, should make the conversion from T implicit
    explicit quaternion(T a = T(), T b = T(), T c = T(), T d = T())
      : base_type(a,b,c,d) { }

    //! Type-conversion
    template <typename X>
    quaternion(const quaternion<X>& q) : base_type(q) { }

    //! Vector initialization
    template <typename E>
    explicit quaternion(const ublas::vector_expression<E>& e) {
      switch (e().size()) {
      case 3: (*this) = quaternion(0, e()(0), e()(1), e()(2)); break;
      case 4: sill::copy(e(), this->begin()); break;
      default: assert(false);
      }
    }

    /**
     * Converts a rotation matrix to a quaternion.
     * A simple algorithm that uses the fact that the rotation matrix
     * must have an eigenvector with eigenvalue 1; this vector is the
     * rotation axis. This algorithm may not be super-stable.
     */
    template <typename E>
    explicit quaternion(const ublas::matrix_expression<E>& e) {
      using namespace std; // math functions will be found in STL or by ADL
      const E& a = e();
      assert(a.size1() == 3 && a.size2() == 3);
      vector3 v(a(2,1) - a(1,2), a(0,2) - a(2,0), a(1,0) - a(0,1));
      T norm = norm_2(v);
      if (norm == 0) {
        throw std::invalid_argument
          ("e is not a rotation matrix or has multiple axes of rotation.");
      }
      v /= norm;
      T alpha = acos((sum(diag(a)) - 1) / 2);
      this->a = cos(alpha/2);
      this->b = sin(alpha/2) * v(0);
      this->c = sin(alpha/2) * v(1);
      this->d = sin(alpha/2) * v(2);
    }

    //! Base type conversion
    quaternion(const base_type& q) : base_type(q) { }

    base_type& base() {
      return static_cast<base_type&>(*this);
    }

    const base_type& base() const {
      return static_cast<const base_type&>(*this);
    }

    //! Assignment (need b/c vector_expression makes operator=
    //! intentionally private).
    quaternion& operator=(const quaternion& q) {
      base_type::operator=(q);
      return *this;
    }

    // Quaternion functions
    //! multiplies the quaternion by a constant s.t. norm_2(*this) == 1
    quaternion& normalize() {
      T norm = norm_2(*this);
      assert(norm > 0);
      (*this) /= norm;
      return *this;
    }

    //! rotates a 3D point by this quaternion
    template <typename E>
    vector3 operator()(const ublas::vector_expression<E>& e) const {
      assert(e().size() == 3);
      return quaternion((*this) * quaternion(e()) * conj(*this)).point();
    }

    //! Computes the derivative of quaternion rotation wrt this quaternion
    //! Assumes that the the quaternion is normalized.
    template <typename E>
    ublas::fixed_matrix<T,3,4>
    drotate(const ublas::vector_expression<E>& v) const {
      assert(v().size()==3);
      T q0 = this->a, q1 = this->b, q2 = this->c, q3 = this->d;
      typename E::value_type v0 = v()(0), v1 = v()(1), v2 = v()(2);
      ublas::fixed_matrix<T,3,4> w;
      w(0,0) = 4*q0*v0 - 2*q3*v1 + 2*q2*v2;
      w(0,1) = 4*q1*v0 + 2*q2*v1 + 2*q3*v2;
      w(0,2) =         + 2*q1*v1 + 2*q0*v2;
      w(0,3) =         - 2*q0*v1 + 2*q1*v2;

      w(1,0) = 2*q3*v0 + 4*q0*v1 - 2*q1*v2;
      w(1,1) = 2*q2*v0           - 2*q0*v2;
      w(1,2) = 2*q1*v0 + 4*q2*v1 + 2*q3*v2;
      w(1,3) = 2*q0*v0           + 2*q2*v2;

      w(2,0) =-2*q2*v0 + 2*q1*v1 + 4*q0*v2;
      w(2,1) = 2*q3*v0 + 2*q0*v1;
      w(2,2) =-2*q0*v0 + 2*q3*v1;
      w(2,3) = 2*q1*v0 + 2*q2*v1 + 4*q3*v2;
      return w;
    }

    //! Computes the derivative of the quaternion normalization function
    ublas::fixed_matrix<T,4,4>
    dnormalize() const {
      double qnorm = norm_2(*this);
      return ublas::identity_matrix<T>(4) / qnorm
        - outer_prod(*this,*this) / pow(qnorm,3);
    }

    //! returns true if this quaternion represents a point
    bool is_point() const {
      using namespace std; // will find abs() in STD or by ADL
      T tol = 1e-10; // precision (assumes we lost 1/2 digits)
      return (abs(this->a) <= sup(*this) * tol) || (abs(this->a) < 1e-100);
    }

    //! converts this quaternion to a point
    //! \requires q(0) \approx 0
    vector3 point() const {
      assert(is_point());
      return vector3(this->b, this->c, this->d);
    }

    // Vector expression interface
    const_iterator begin() const { return const_iterator(*this, 0); }

    iterator begin() { return iterator(*this, 0); }

    const_iterator end() const {
      assert((&(this->a))+3 == &(this->d));
      return const_iterator(*this, 4);
    }

    iterator end() {
      assert((&(this->a))+3 == &(this->d));
      return iterator(*this, 4);
    }

    const_iterator find (size_type i) const {
      return const_iterator(*this, i);
    }

    iterator find (size_type i) {
      return iterator(*this, i);
    }

    //! number of elements (4)
    size_type size() const { return 4; }

    void swap(quaternion& other) { std::swap(*this, other); }

    //! element access
    const T& operator()(size_type i) const {
      assert(i<4); return *((&(this->a))+i);
    }

    //! mutable element access
    T& operator()(size_type i) { assert(i<4); return *((&(this->a))+i); }

    //! element access
    T operator[](size_type i) const { return operator()(i); }

    //! mutable element access
    T& operator[](size_type i) { return operator()(i); }

    //! assignment without aliasing
    template <typename E>
    quaternion& assign(const ublas::vector_expression<E>& e) {
      sill::copy(e(), begin());
      return *this;
    }

    //! assignment with a temporary
    template <typename E>
    quaternion& operator=(const ublas::vector_expression<E>& e) {
      return (*this = quaternion(e()));
    }

    //! computed assignment without aliasing
    template <typename E>
    quaternion& plus_assign(const ublas::vector_expression<E>& e) {
      assert(e().size() == 4);
      this->a+=e()(0); this->b+=e()(1); this->c+=e()(2); this->d+=e()(3);
      return *this;
    }

    //! computed assignment without aliasing
    template <typename E>
    quaternion& minus_assign(const ublas::vector_expression<E>& e) {
      assert(e().size() == 4);
      this->a-=e()(0); this->b-=e()(1); this->c-=e()(2); this->d-=e()(3);
      return *this;
    }

    //! computed assignment with a temporary
    template <typename E>
    quaternion& operator+=(const ublas::vector_expression<E>& e) {
      return plus_assign(quaternion(e()));
    }

    //! computed assignment with a temporary
    template <typename E>
    quaternion& operator-=(const ublas::vector_expression<E>& e) {
      return minus_assign(quaternion(e()));
    }
  }; // class quaternion

  // overloads to resolve ambiguity with vector_expression
  //! \relates quaternion
  template <typename T>
  quaternion<T> conj(const quaternion<T>& q) {
    return conj(q.base());
  }

  //! \relates quaternion
  template <typename T>
  bool operator==(const quaternion<T>& q1, const quaternion<T>& q2) {
    return q1.base() == q2.base();
  }

  // overloads to get the right return type
  //! \relates quaternion
  template <typename T>
  quaternion<T> operator*(const quaternion<T>& a, const quaternion<T>& b) {
    return a.base() * b.base();
  }

  //! \relates quaternion
  template <typename T>
  quaternion<T> operator+(const quaternion<T>& a, const quaternion<T>& b) {
    return a.base() + b.base();
  }

  //! \relates quaternion
  template <typename T>
  quaternion<T> operator-(const quaternion<T>& a, const quaternion<T>& b) {
    return a.base() - b.base();
  }

  // TODO: other operators

  //! \relates quaternion
  template <typename Char, typename Traits, typename T>
  std::basic_ostream<Char,Traits>&
  operator<<(std::basic_ostream<Char,Traits>& out, const quaternion<T>& q) {
    out << static_cast<const boost::math::quaternion<T>&>(q);
    return out;
  }

  // @} group math_ublas

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

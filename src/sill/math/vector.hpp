#ifndef SILL_VECTOR_HPP
#define SILL_VECTOR_HPP

#warning "sill/math/vector.hpp is deprecated"

#include <functional>
#include <ostream>
#include <string>
#include <vector>

#include <itpp/base/vec.h>
#include <itpp/base/mat.h>
#include <itpp/base/matfunc.h>
#include <sill/math/irange.hpp>
#include <sill/stl_concepts.hpp>
#include <sill/range/forward_range.hpp>
#include <sill/serialization/serialize.hpp>
#include <sill/macros_def.hpp>

namespace sill {

  // Pre-declaration
  template <typename Ref> class forward_range;

  // We reimplement some of the functions, to maintain a complete
  // documentation

  /**
   * A dense vector.
   * This class provides standardized API for the IT++ vector class
   * and extends its functionality to better work with subvectors.
   * @see http://itpp.sourceforge.net/current/classitpp_1_1Vec.html
   *
   * Note that most non-trivial indexing functions (such as row() and
   * operator()(irange, irange)) return a copy of the corresponding
   * elements. This convention differs from Matlab, where one can
   * write a([1 2 3],[1 2 3]) = 3 (thus, in Matlab, indexing creates
   * a reference to a submatrix). To assign or manipulate submatrices,
   * use the functions set_row, set_submatrix, add_row, etc.
   *
   * All indexing is 0-based. The vector can be used in STL algorithms.
   *
   * \ingroup math_linalg
   */
  template <typename T>
  class vector : public itpp::Vec<T> {
  
  #ifdef _MSC_VER  
    // in VC++, itpp::Vec::T() shadows the template argument T
    typedef value_type T;
  #endif

    // Public type declarations
    //==========================================================================
  public:
    // Container interface
    typedef T               value_type;
    typedef std::ptrdiff_t  difference_type;
    typedef size_t          size_type;
    typedef T&              reference;
    typedef T*              pointer;
    typedef T*              iterator;
    typedef const T&        const_reference;
    typedef const T*        const_pointer;
    typedef const T*        const_iterator;

    // The base type
    typedef itpp::Vec<T> base;

    // Constructors
    //==========================================================================
  public:

    //! Default constructor
    vector() : base() {  }

    //! Constructs a vector with the specified dimension
    //! Note: the vector is not necessarily zero-initialized
    explicit vector(size_t size) : base(size) {
      assert(size <= (size_t)(std::numeric_limits<int>::max()));
    }

    //! Constructs a vector filled with the given element
    vector(size_t size, T value) : base(size) {
      assert(size <= (size_t)(std::numeric_limits<int>::max()));
      std::fill_n(base::data, size, value);
    }

    //! Conversion from the base class
    vector(const base& v) : base(v) { }

    //! Creates a vector from a C array (the contents of the array are copied).
    vector(const T* array, size_t size) : base(array, size) {
      assert(size <= (size_t)(std::numeric_limits<int>::max()));
    }

    //! Creates a vector from an STL vector (the contents are copied)
    vector(const std::vector<T>& v) : base(&v[0], v.size()) {
      assert(v.size() <= (size_t)(std::numeric_limits<int>::max()));
    }

    //! Creates a vector from an STL vector of a different type
    //! (the contents are copied)
    template <typename U>
    vector(const std::vector<U>& v) : base(v.size()) {
      assert(v.size() <= (size_t)(std::numeric_limits<int>::max()));
      for(size_t i = 0; i < size(); i++)
        base::set(i, v[i]);
    }

    //! Conversion from human-readable representation
    vector(const char* str) : base(str) { }

    //! Conversion from human-readable representation
    vector(const std::string& str) : base(str) { }

    //! Conversion from a different vector
    template <typename U>
    vector(const itpp::Vec<U>& v) : base(v.size()) {
      for(size_t i = 0; i < size(); i++)
        base::set(i, v[i]);
    }

    //! Swap with another matrix.
    //! @todo Make this more efficient!
    void swap(vector& other) {
      vector tmp(other);
      other = *this;
      *this = tmp;
    }

    // Range interface
    //==========================================================================
    //! Returns a pointer to the first element
    const T* begin() const {
      return base::_data();
    }

    //! Returns a pointer to the first element
    T* begin() {
      return base::_data();
    }

    //! Returns a pointer past the last element
    const T* end() const {
      return begin() + size();
    }

    //! Returns a pointer past the last element
    T* end() {
      return begin() + size();
    }

    // Accessors
    //==========================================================================
    //! Returns the size of the vector
    size_t size() const {
      return base::datasize;
    }

    //! Returns true if the vector has no elements
    bool empty() const {
      return base::datasize == 0;
    }

    //! Changes the size of the vector.
    //! \param copy if true, retains the original data
    void resize(size_t size, bool copy = false) {
      base::set_size(size, copy);
    }

    //! Returns the i-th element.
    const T& operator[](size_t i) const {
      return base::operator[](i);
    }

    //! Returns the i-th element.
    const T& operator()(size_t i) const {
      return base::operator()(i);
    }

    //! Returns the i-th element.
    T& operator[](size_t i) {
      return base::operator[](i);
    }

    //! Returns the i-th element.
    T& operator()(size_t i) {
      return base::operator()(i);
    }

    //! Returns a continuous range of the vector
    const itpp::Vec<T> operator()(irange i) const {
      if (i.empty())
        return itpp::Vec<T>(0);
      else
        return base::operator()(i.start(), i.end());
    }

    //! Returns a sub-vector with indices given by \c i
    const itpp::Vec<T> operator()(const itpp::uvec& i) const {
      return base::operator()(i);
    }

    //! Returns the elements for which \c i is \c 1
    const itpp::Vec<T> operator()(const itpp::bvec& i) const {
      return base::operator()(i);
    }

    //! Returns the right (last?) \c nr elements of the vector
    itpp::Vec<T> right(size_t n) const {
      return base::right(n);
    }

    //! Returns the left (first?) \c nr elements of the vector
    itpp::Vec<T> left(size_t n) const {
      return base::left(n);
    }

    //! Returns \c n elements of the vector starting from \c start
    itpp::Vec<T> middle(size_t start, size_t n) const {
      return base::middle(start, n);
    }

    //! Returns the transpose of this vector
    itpp::Mat<T> transpose() const {
      return base::transpose();
    }

    //! Returns the conjugate transpose of this vector
    itpp::Mat<T> hermitian_transpose() {
      return base::hermitian_transpose();
    }

    // Comparison operators
    //==========================================================================
    //! Element-wise comparison with a scalar
    itpp::bvec operator==(T value) const {
      return base::operator==(value);
    }

    //! Element-wise comparison with a scalar
    itpp::bvec operator!=(T value) const {
      return base::operator!=(value);
    }

    //! Element-wise comparison with a scalar
    itpp::bvec operator<(T value) const {
      return base::operator<(value);
    }

    //! Element-wise comparison with a scalar
    itpp::bvec operator<=(T value) const {
      return base::operator<=(value);
    }

    //! Element-wise comparison with a scalar
    itpp::bvec operator>(T value) const {
      return base::operator>(value);
    }

    //! Element-wise comparison with a scalar
    itpp::bvec operator>=(T value) const {
      return base::operator>=(value);
    }

    //! Compares the vectors (for use with ordered sets), first by comparing
    //! lengths and then by comparing elements in order.
    bool operator<(const vector& v) const {
      if (size() < v.size()) {
        return true;
      } else if (size() > v.size()) {
        return false;
      } else {
        for (size_t i(0); i < size(); ++i) {
          if (operator[](i) < v[i])
            return true;
          else if (operator[](i) > v[i])
            return false;
        }
        return false;
      }
    }

    //! Returns true if the vectors have the same length and values
    bool operator==(const itpp::Vec<T>& v) const {
      return base::operator==(v);
    }

    //! Returns true if the vectors have different length or values.
    bool operator!=(const itpp::Vec<T> &v) const {
      return base::operator!=(v);
    }

    // Modifiers
    //==========================================================================
    //! Sets all the elements of the vector to \c value
    vector& operator=(T value) {
      base::operator=(value); return *this;
    }

    //! Assigns a vector
    vector& operator=(const itpp::Vec<T>& v) {
      base::operator=(v); return *this;
    }

    //! Assign vector the values in the string \c values
    vector& operator=(const char* str) {
      base::operator=(str); return *this;
    }

    //! Sets all entries of to zero
    void clear() {
      base::zeros();
    }

    //! Sets all entries of to zero
    void zeros() {
      base::zeros();
    }

    //! Sets all entries to zero using memset.
    //! (Joseph: I kept this separate, but feel free to replace zeros()
    //!  with this.)
    void zeros_memset() {
      memset(base::_data(), 0, sizeof(T) * base::size());
    }

    //! Sets all entries of to one
    void ones() {
      base::ones();
    }

    //! Sets a continuous subrange to a vector
    void set_subvector(irange i, const vector<T>& v) {
      assert(i.size() == v.size());
      assert(i.stop() <= size());
      for (size_t k(0); k < i.size(); ++k) {
        this->operator[](i.start() + k) = v[k];
      }
      /* // REPLACED SINCE THIS set_subvector SYNTAX IS DEPRECATED IN IT++.
      if (!i.empty())
        base::set_subvector(i.start(), i.end(), v);
      */
    }

    //! Sets a continuous subrange to a value
    void set_subvector(irange i, T value) {
      assert(i.stop() <= size());
      for (size_t k(0); k < i.size(); ++k) {
        this->operator[](i.start() + k) = value;
      }
      /* // REPLACED SINCE THIS set_subvector SYNTAX IS DEPRECATED IN IT++.
      if (!i.empty())
        base::set_subvector(i.start(), i.end(), value);
      */
    }

    //! Sets a subvector to a vector
    void set_subvector(const itpp::uvec& i, const itpp::Vec<T>& v) {
      assert(i.size() == v.size());
      for(int k = 0; k < i.size(); k++)
        base::set(i(k), v(k));
    }

    //! Sets a subvector to a vector
    void set_subvector(const itpp::uvec& i, T value) {
      for(size_t k = 0; k < i.size(); k++)
        base::set(i(k), value);
    }

    //! Removes the i-th element
    void remove(size_t i) {
      base::del(i);
    }

    //! Removes the elements in a continuous range
    void remove(size_t i1, size_t i2) {
      base::del(i1, i2);
    }

    //! Inserts an element at position \c i
    void insert(size_t i, T value) {
      base::ins(i, value);
    }

    //! Inserts a vector at position \c i
    void insert(size_t i, const itpp::Vec<T>& v) {
      base::ins(i, v);
    }

    //! Flips the vector in place 
    //! (i.e., reorders the elements from end to the beginning)
    void flip() {
      size_t n = size();
      for(size_t i = 0; i < n / 2; i++)
        std::swap(base::_elem(i), base::_elem(n-i-1));
    }

  /*
    //! Shift in element \c In at position 0 \c n times
    void shift_right(const Num_T In, int n=1);

    //! Shift in vector \c In at position 0
    void shift_right(const Vec<Num_T> &In);

    //! Shift out the \c n left elements and a the same time shift in the element \c at last position \c n times
    void shift_left(const Num_T In, int n=1);

    //! Shift in vector \c In at last position
    void shift_left(const Vec<Num_T> &In);

    //! Splits the vector. Returns the first part and retains the second.
    itpp::Vec<T> split(size_t pos) const {
      return base::split(pos);
    }

  */

    //! Updates a subvector with the specified binary function
    template <typename F>
    void update_subvector(irange i, const itpp::Vec<T>& v, F f) {
      concept_assert((BinaryFunction<F, T, T, T>));
      assert(i.size() == v.size());
      for(size_t k = 0; k < v.size(); k++)
        base::set(i(k), f(operator()(i(k)), v[k]));
    }

    //! Updates a subvector with the specified binary function
    template <typename F>
    void update_subvector(const itpp::uvec i, const itpp::Vec<T>& v, F f) {
      concept_assert((BinaryFunction<F, T, T, T>));
      assert(i.size() == v.size());
      for(int k = 0; k < v.size(); k++)
        base::set(i(k), f(operator()(i(k)), v[k]));
    }

    // Arithmetic operations
    //==========================================================================

    //! Vector addition
    vector& operator+=(const itpp::Vec<T>& v) {
      base::operator+=(v); return *this;
    }

    //! Adds a scalar
    vector& operator+=(T value) {
      base::operator+=(value); return *this;
    }

    //! Vector subtraction
    vector& operator-=(const itpp::Vec<T>& v) {
      base::operator-=(v); return *this;
    }

    //! Subtracts a scalar
    vector& operator-=(T value) {
      base::operator-=(value); return *this;
    }

    //! Multiplies by a scalar
    vector& operator*=(T value) {
      base::operator*=(value); return *this;
    }
    
    //! Element-wise multiplication
    vector& operator*=(const itpp::Vec<T>& v) {
      elem_mult_inplace(v, *this); return *this;
    }

    //! Element-wise division
    vector& operator/=(T value) {
      base::operator/=(value); return *this;
    }

    //! Element-wise division
    vector& operator/=(const itpp::Vec<T>& v) {
      base::operator/=(v); return *this;
    }

    //! Adds to a subvector of this vector
    void add_subvector(irange i, const itpp::Vec<T>& v) {
      update_subvector(i, v, std::plus<T>());
    }

    //! Adds to a subvector of this vector
    void add_subvector(const itpp::uvec& i, const itpp::Vec<T>& v) {
      update_subvector(i, v, std::plus<T>());
    }

    //! Subtracts from a subvector of this vector
    void subtract_subvector(irange i, const itpp::Vec<T>& v) {
      update_subvector(i, v, std::minus<T>());
    }

    //! Subtracts from a subvector of this vector
    void subtract_subvector(const itpp::uvec& i, const itpp::Vec<T>& v) {
      update_subvector(i, v, std::minus<T>());
    }

    //! Multiplies a subvector of this vector element-wise
    //! \todo Fix the name?
    void multiply_subvector(irange i, const itpp::Vec<T>& v) {
      update_subvector(i, v, std::multiplies<T>());
    }

    //! Multiplies a subvector of this vector element-wise
    //! \todo Fix the name?
    void multiply_subvector(const itpp::uvec& i, const itpp::Vec<T>& v) {
      update_subvector(i, v, std::multiplies<T>());
    }

    //! Divides a subvector of this vector element-wise
    void divide_subvector(irange i, const itpp::Vec<T>& v) {
      update_subvector(i, v, std::divides<T>());
    }

    //! Divides a subvector of this vector element-wise
    void divide_subvector(const itpp::uvec& i, const itpp::Vec<T>& v) {
      update_subvector(i, v, std::divides<T>());
    }

    // Methods to support optimization routines
    //==========================================================================

    //! Inner product
    T inner_prod(const itpp::Vec<T>& x) const {
      return dot(*this, x);
    }

    //! Element-wise multiplication with another value of the same size.
    vector& elem_mult(const vector& other) {
      elem_mult_inplace(other, *this);
      return *this;
    }

    //! Element-wise reciprocal (i.e., change v to 1/v).
    vector& reciprocal() {
      foreach(double& val, *this) {
        assert(val != 0);
        val = 1./val;
      }
      return *this;
    }

    //! Returns the L1 norm.
    double L1norm() const {
      return sum(abs(*this));
    }

    //! Returns the L2 norm.
    double L2norm() const {
      return sqrt(inner_prod(*this));
    }

    //! Returns a vector of the same size but with values replaced by their
    //! signs (-1 for negative, 0 for 0, 1 for positive).
    vector sign() const {
      vector v(*this);
      foreach(double& val, v)
        val = (val > 0 ? 1 : (val == 0 ? 0 : -1) );
      return v;
    }

    //! Print info about this vector (for debugging).
    void print_info(std::ostream& out) const {
      out << "vec size: " << size() << "\n";
    }

    // Serialization
    //==========================================================================

    void save(oarchive& ar) const {
      ar << size();
      for(size_t i = 0; i < size(); i++) {
        ar << (operator[](i));
      }
    }
    
    void load(iarchive& ar) {
      size_t n;
      ar >> n;
      resize(n);
      for(size_t i = 0; i < size(); i++) {
        ar >> (operator[](i));
      }
    }

    /**
     * Print in human-readable format.
     * Separate each element with elem_delimiter (default = space),
     * and end with end_delimiter (default = newline).
     */
    void print(std::ostream& out, const std::string& elem_delimiter = " ",
               const std::string& end_delimiter = "\n") const {
      for (size_t i(0); i < size(); ++i) {
        out << operator()(i);
        if (i + 1 == size())
          out << end_delimiter;
        else
          out << elem_delimiter;
      }
    }

  }; // class vector


  // Standardized free functions
  //============================================================================

  //! Vector addition
  //! \relates vector
  template <typename T>
  vector<T> operator+(const vector<T>& v1, const vector<T>& v2) {
    return (itpp::Vec<T>(v1) + itpp::Vec<T>(v2));
  }

  //! Vector subtraction
  //! \relates vector
  template <typename T>
  vector<T> operator-(const vector<T>& v1, const vector<T>& v2) {
    return (itpp::Vec<T>(v1) - itpp::Vec<T>(v2));
  }

  //! Vector multiplication by a scalar.
  //! \relates vector
  template <typename T>
  vector<T> operator*(const vector<T>& v, double d) {
    return (itpp::Vec<T>(v) * d);
  }

  //! Vector multiplication by a scalar.
  //! \relates vector
  template <typename T>
  vector<T> operator*(const vector<T>& v, int d) {
    return (itpp::Vec<T>(v) * d);
  }

  //! Vector multiplication by a scalar.
  //! \relates vector
  template <typename T>
  vector<T> operator*(double d, const vector<T>& v) {
    return (d * itpp::Vec<T>(v));
  }

  //! Vector multiplication by a scalar.
  //! \relates vector
  template <typename T>
  vector<T> operator*(int d, const vector<T>& v) {
    return (d * itpp::Vec<T>(v));
  }

  //! Inner product
  //! \relates vector
  template <typename T>
  T inner_prod(const itpp::Vec<T>& x, const itpp::Vec<T>& y) {
    return dot(x, y);
  }

  //! Element-wise multiplication of two vectors of the same size.
  //! \relates vector
  template <typename T>
  vector<T> elem_mult(const vector<T>& x, const vector<T>& y) {
    return itpp::elem_mult(itpp::Vec<T>(x), itpp::Vec<T>(y));
  }

  //! \relates matrix
  template <typename T>
  itpp::Mat<T> trans(const itpp::Vec<T>& a) {
    return a.transpose();
  }

  //! \relates matrix
  template <typename T>
  itpp::Mat<T> herm(const itpp::Vec<T>& a) {
    return a.hermitian_transpose();
  }

  //! Concatenates a sequence of vectors
  vector<double> concat(const forward_range<const vector<double>&> vectors);

  // Type definitions
  //============================================================================

  //! \relates vector
  typedef vector<double> vec;

  //! \relates vector
  typedef vector<std::complex<double> > cvec;

  //! \relates vector
  typedef vector<int> uvec;

  //! \relates vector
  typedef vector<short> svec;

  //! \relates vector
  typedef vector<itpp::bin> bvec;

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

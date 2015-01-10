#ifndef SILL_GAUSSIAN_OPT_VECTOR_HPP
#define SILL_GAUSSIAN_OPT_VECTOR_HPP

#include <armadillo>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Type which parametrizes Gaussian factors, usable for optimization and
   * learning. This object consists of:
   * A (n x n matrix), b (n vector), C (n x m matrix)
   *
   * \see gaussian_crf_factor
   * \ingroup optimization_classes
   */
  template <typename T>
  struct gaussian_opt_vector {
    
    //! The type of values stored in this vector
    typedef T value_type;

    //! The type that represents a matrix of parameters
    typedef arma::Mat<T> mat_type;

    //! The type that represents a vector of parameters
    typedef arma::Col<T> vec_type;

    //! Size n x n
    mat_type a;

    //! Size n
    vec_type b;

    //! Size n x m
    mat_type c;

    // Constructors and serialization
    //========================================================================
    gaussian_opt_vector() { }

    gaussian_opt_vector(size_t nhead, size_t ntail, T val = 0.0)
      : a(nhead, nhead), b(nhead), c(nhead, ntail) {
      a.fill(val);
      b.fill(val);
      c.fill(val);
    }

    gaussian_opt_vector(const mat_type& a, const vec_type& b, const mat_type& c)
      : a(a), b(b), c(c) {
      if (c.n_rows == 0 && c.n_cols == 0 && a.n_rows != 0) {
        this->c.set_size(a.n_rows, 0);
      }
      if (!valid_size()) {
        throw std::invalid_argument(
          "gaussian_opt_vector: dimensions do not match each other"
        );
      }
    }

    //! Serialize members
    void save(oarchive & ar) const {
      ar << a << b << c;
    }

    //! Deserialize members
    void load(iarchive & ar) {
      ar >> a >> b >> c;
    }

    // Size operations
    //========================================================================
    //! Returns the number of head variables
    size_t nhead() const {
      return a.n_rows;
    }

    //! Returns the number of tail variables
    size_t ntail() const {
      return c.n_cols;
    }

    //! Returns true iff the sizes of a, b, c match each other.
    bool valid_size() const {
      return
        (a.n_rows == a.n_cols) &&
        (a.n_rows == b.n_rows) &&
        (a.n_rows == c.n_rows);
    }

    //! Resize the data.
    void resize(size_t nhead, size_t ntail) {
      a.set_size(nhead, nhead);
      b.set_size(nhead);
      c.set_size(nhead, ntail);
    }

    // Vector operations
    //========================================================================

    //! Adds another vector to this one
    gaussian_opt_vector& operator+=(const gaussian_opt_vector& other) {
      a += other.a;
      b += other.b;
      c += other.c;
      return *this;
    }

    //! Subtracts another vector from this one
    gaussian_opt_vector& operator-=(const gaussian_opt_vector& other) {
      a -= other.a;
      b -= other.b;
      c -= other.c;
      return *this;
    }

    //! Element-wise multiplication by another vector
    gaussian_opt_vector& operator%=(const gaussian_opt_vector& other) {
      a %= other.a;
      b %= other.b;
      c %= other.c;
      return *this;
    }

    //! Multiplication by a scalar value.
    gaussian_opt_vector& operator*=(T x) {
      a *= x;
      b *= x;
      c *= x;
      return *this;
    }

    //! ELement-wise division by another vector
    gaussian_opt_vector& operator/=(const gaussian_opt_vector& other) {
      a /= other.a;
      b /= other.b;
      c /= other.c;
      return *this;
    }

    //! Division by a scalar value.
    gaussian_opt_vector& operator/=(T x) {
      a /= x;
      b /= x;
      c /= x;
      return *this;
    }

    /**
     * "Zeros" this vector by setting b, C to 0 and setting A to be a
     * diagonal matrix (identity by default).
     * @param diag  For this to be a "zero" factor which exerts little
     *              influence on the model, A needs to be diagonal with
     *              very small entries.  This method sets A's diagonal to
     *              this value.
     *              (default = 1)
     */
    void zeros(T diag = 1.0) {
      a = diag * arma::eye<mat_type>(a.n_rows, a.n_rows);
      b.zeros();
      c.zeros();
    }

    //! Print info about this vector (for debugging).
    void print_info(std::ostream& out) const {
      out << "gaussian_opt_vector(" << nhead() << ", " << ntail() << ")";
    }

  }; // struct gaussian_opt_vector

  // Free functions
  //========================================================================

  //! Outputs the parameters to a stream
  template <typename T>
  std::ostream& operator<<(std::ostream& out, const gaussian_opt_vector<T>& x) {
    out << "A:\n" << x.a
        << "b:\n" << x.b
        << "C:\n" << x.c;
    return out;
  }

  //! Returns true iff this instance equals the other.
  template <typename T>
  bool operator==(const gaussian_opt_vector<T>& x,
                  const gaussian_opt_vector<T>& y) {
    return equal(x.a, y.a) && equal(x.b, y.b) && equal(x.c, y.c);
  }
  
  //! Returns false iff this instance equals the other.
  template <typename T>
  bool operator!=(const gaussian_opt_vector<T>& x,
                  const gaussian_opt_vector<T>& y) {
    return !(x == y);
  }

  //! Adds a scalar multiple of a vector to another vector
  template <typename T>
  void axpy(T a, const gaussian_opt_vector<T>& x, gaussian_opt_vector<T>& y) {
    y.a += a * x.a;
    y.b += a * x.b;
    y.c += a * x.c;
  }

  //! Returns a vector whose each element is equal to the sign of the corresponding 
  //! element in the input vector (-1 for negative, 0 for 0, 1 for positive).
  template <typename T>
  gaussian_opt_vector<T> sign(const gaussian_opt_vector<T>& x) {
    return gaussian_opt_vector(sign(x.a), sign(x.b), sign(x.c));
  }

  //! Inner product of two vectors
  template <typename T>
  T dot(const gaussian_opt_vector<T>& x, const gaussian_opt_vector<T>& y) {
    return dot(x.a, y.a) + dot(x.b, y.b) + dot(x.c, y.c);
  }

  //! Returns the L2 norm of a vector
  template <typename T>
  T norm_2(const gaussian_opt_vector<T>& x) {
    return std::sqrt(dot(x, x));
  }

}  // namespace sill

#include <sill/macros_undef.hpp>

#endif

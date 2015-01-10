#ifndef SILL_LOGREG_OPT_VECTOR_HPP
#define SILL_LOGREG_OPT_VECTOR_HPP

#include <armadillo>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Optimization variables (which fit the OptimizationVector concept)
   * for multiclass logistic regression.
   *
   * \tparam T   Type of floating point number.
   */
  template <typename T>
  struct logreg_opt_vector {

    //! The type of values stored in this vector
    typedef T value_type;

    //! The type that represents a matrix of parameters
    typedef arma::Mat<T> mat_type;

    //! The type that represents a vector of parameters
    typedef arma::Col<T> vec_type;

    //! f(k,j) = weight for label k, finite index j
    mat_type f;

    //! v(k,j) = weight for label k, vector index j
    mat_type v;

    //! Offsets b; b(k) = offset for label k
    vec_type b;

    // Constructors
    //========================================================================
    logreg_opt_vector() { }

    logreg_opt_vector(size_t num_labels,
                      size_t num_finite,
                      size_t num_vector,
                      T init = 0)
      : f(num_labels, num_finite), v(num_labels, num_vector), b(num_finite) {
      f.fill(init);
      v.fill(init);
      b.fill(init);
    }

    logreg_opt_vector(const mat_type& f,
                      const mat_type& v,
                      const vec_type& b)
      : f(f), v(v), b(b) {
      assert(f.n_rows == v.n_rows && v.n_rows == b.n_rows);
    }

    // Size operations
    //========================================================================
    //! Returns the number of labels
    size_t nlabels() const {
      return b.n_rows;
    }

    //! Returns the number of finite tail indices
    size_t nfinite() const {
      return f.n_cols;
    }

    //! Returns the number of vector tail indices
    size_t nvector() const {
      return v.n_cols;
    }

    //! Resize the data.
    void resize(size_t num_labels,
                size_t num_finite,
                size_t num_vector) {
      f.set_size(num_labels, num_finite);
      v.set_size(num_labels, num_vector);
      b.set_size(num_labels);
    }

    // Vector operations
    //========================================================================
    //! Adds another vector to this one.
    logreg_opt_vector& operator+=(const logreg_opt_vector& other) {
      f += other.f;
      v += other.v;
      b += other.b;
      return *this;
    }

    //! Adds a scalar value to this vector
    logreg_opt_vector& operator+=(T x) {
      f += x;
      v += x;
      b += x;
      return *this;
    }

    //! Subtracts another vector from this one.
    logreg_opt_vector& operator-=(const logreg_opt_vector& other) {
      f -= other.f;
      v -= other.v;
      b -= other.b;
      return *this;
    }

    //! Subtracts a scalar value from this vector
    logreg_opt_vector& operator-=(T x) {
      f -= x;
      v -= x;
      b -= x;
      return *this;
    }

    //! Element-wise multiplication by another parameter vector
    logreg_opt_vector& operator%=(const logreg_opt_vector& other) {
      f %= other.f;
      v %= other.v;
      b %= other.b;
      return *this;
    }

    //! Multiplication by a scalar value.
    logreg_opt_vector& operator*=(T x) {
      f *= x;
      v *= x;
      b *= x;
      return *this;
    }

    //! Element-wise division by another parameter vector
    logreg_opt_vector& operator/=(const log_reg_opt_vector& other) {
      f /= other.f;
      v /= other.v;
      b /= other.b;
      return *this;
    }

    //! Division by a scalar value.
    logreg_opt_vector& operator/=(T x) {
      f /= x;
      v /= x;
      b /= x;
      return *this;
    }

    //! Sets all values to 0.
    void zeros() {
      f.zeros();
      v.zeros();
      b.zeros();
    }

    //! Print info about this vector (for debugging).
    void print_info(std::ostream& out) const {
      out << "logreg_opt_vector("
          << nlabels() << ", "
          << nfinite() << ", "
          << nvector() << ")";
    }

    //! Print info about extrema in this vector (for debugging).
    void print_extrema_info(std::ostream& out) const {
      for (size_t i = 0; i < nlabels(); ++i) {
        if (f.size() > 0)
          out << "f(" << i << ",min) = " << min(f.row(i))
              << "\t"
              << "f(" << i << ",max) = " << max(f.row(i))
              << std::endl;
        if (v.size() > 0)
          out << "v(" << i << ",min) = " << min(v.row(i))
              << "\t"
              << "v(" << i << ",max) = " << max(v.row(i))
              << std::endl;
      }
      if (b.size() > 0)
        out << "b(min) = " << b[min_index(b)] << "\t"
            << "b(max) = " << b[max_index(b)] << std::endl;
    }

  }; // struct logreg_opt_vector

  // Free functions
  //========================================================================

  //! Outputs the parameters to a stream
  template <typename T>
  std::ostream& operator<<(std::ostream& out, const logreg_opt_vector<T>& x) {
    out << "f:\n" << x.f
        << "v:\n" << x.v
        << "b:\n" << x.b;
    return out;
  }

  //! Returns true iff two vector are equal
  template <typename T>
  bool operator==(const logreg_opt_vector<T>& x, const logreg_opt_vector<T>& y) {
    return equal(x.f, y.f) && equal(x.v, y.v) && equal(x.b, y.b);
  }

  //! Returns false iff this instance equals the other.
  template <typename T>
  bool operator!=(const logreg_opt_vector<T>& x, cosnt logreg_opt_vector<T>& y) {
    return !(x == y);
  }

  //! Adds a scalar multiple of a vector to another vector
  template <typename T>
  void axpy(T a, const logreg_opt_vector<T>& x, logreg_opt_vector<T>& y) {
    y.f += a * x.f;
    y.v += a * x.v;
    y.b += a * x.b;
  }

  //! Returns a vector whose each element is equal to the sign of the corresponding 
  //! element in the input vector (-1 for negative, 0 for 0, 1 for positive).
  template <typename T>
  logreg_opt_vector<T> sign(const logreg_opt_vector<T>& x) {
    return logreg_opt_vector(sign(x.f), sign(x.v), sign(x.b));
  }

  //! Inner product of two vectors
  template <typename T>
  T dot(const logreg_opt_vector<T>& x, const logreg_opt_vector<T>& y) {
    return dot(x.f, y.f) + dot(x.v, y.v) + dot(x.b, y.b);
  }

  //! Returns the L1 norm of a vector
  template <typename T>
  T norm_1(const logreg_opt_vector<T>& x) {
    return accu(abs(x.f)) + accu(abs(x.v)) + accu(abs(x.b));
  }

  //! Returns the L2 norm of a vector
  template <typename T>
  T norm_2(const logreg_opt_vector<T>& x) {
    return std::sqrt(dot(x, x));
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

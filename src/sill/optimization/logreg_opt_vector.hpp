#ifndef SILL_LOGREG_OPT_VECTOR_HPP
#define SILL_LOGREG_OPT_VECTOR_HPP

#include <sill/math/linear_algebra/armadillo.hpp>
#include <sill/math/statistics.hpp>
#include <sill/optimization/optimization_vector.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Optimization variables (which fit the OptimizationVector concept)
   * for multiclass logistic regression.
   *
   * @tparam T   Type of floating point number.
   */
  template <typename T>
  struct logreg_opt_vector {

    // Types and data
    //------------------------------------------------------------------------

    typedef T value_type;

    struct size_type {
      size_t f_rows;
      size_t f_cols;
      size_t v_rows;
      size_t v_cols;
      size_t b_size;

      size_type() { }

      size_type(size_t f_rows, size_t f_cols, size_t v_rows, size_t v_cols,
                size_t b_size)
        : f_rows(f_rows), f_cols(f_cols), v_rows(v_rows), v_cols(v_cols),
          b_size(b_size) { }

      bool operator==(const size_type& other) const {
        return ((f_rows == other.f_rows) && (f_cols == other.f_cols) &&
                (v_rows == other.v_rows) && (v_cols == other.v_cols) &&
                (b_size == other.b_size));
      }

      bool operator!=(const size_type& other) const {
        return (!operator==(other));
      }
    };

    //! f(k,j) = weight for label k, finite index j
    mat f;

    //! v(k,j) = weight for label k, vector index j
    mat v;

    //! Offsets b; b(k) = offset for label k
    vec b;

    // Constructors
    //------------------------------------------------------------------------

    logreg_opt_vector() { }

    logreg_opt_vector(size_type s, value_type default_val = 0)
      : f(s.f_rows, s.f_cols), v(s.v_rows, s.v_cols), b(s.b_size) {
      f.fill(default_val);
      v.fill(default_val);
      b.fill(default_val);
    }

    logreg_opt_vector(const mat& f, const mat& v, const vec& b)
      : f(f), v(v), b(b) { }

    // Getters and non-math setters
    //------------------------------------------------------------------------

    //! Returns true iff this instance equals the other.
    bool operator==(const logreg_opt_vector& other) const {
      return (equal(f, other.f) && equal(v, other.v) && equal(b, other.b));
    }

    //! Returns false iff this instance equals the other.
    bool operator!=(const logreg_opt_vector& other) const {
      return !operator==(other);
    }

    size_type size() const {
      return size_type(f.n_rows, f.n_cols, v.n_rows, v.n_cols, b.size());
    }

    //! Resize the data.
    void resize(const size_type& newsize) {
      f.set_size(newsize.f_rows, newsize.f_cols);
      v.set_size(newsize.v_rows, newsize.v_cols);
      b.set_size(newsize.b_size);
    }

    // Math operations
    //------------------------------------------------------------------------

    //! Sets all elements to this value.
    logreg_opt_vector& operator=(value_type d) {
      f.fill(d);
      v.fill(d);
      b.fill(d);
      return *this;        
    }

    //! Addition.
    logreg_opt_vector operator+(const logreg_opt_vector& other) const {
      return logreg_opt_vector(f + other.f, v + other.v, b + other.b);
    }

    //! Addition.
    logreg_opt_vector& operator+=(const logreg_opt_vector& other) {
      f += other.f;
      v += other.v;
      b += other.b;
      return *this;
    }

    //! Subtraction.
    logreg_opt_vector operator-(const logreg_opt_vector& other) const {
      return logreg_opt_vector(f - other.f, v - other.v, b - other.b);
    }

    //! Subtraction.
    logreg_opt_vector& operator-=(const logreg_opt_vector& other) {
      f -= other.f;
      v -= other.v;
      b -= other.b;
      return *this;
    }

    //! Scalar subtraction.
    logreg_opt_vector operator-(value_type d) const {
      return logreg_opt_vector(f - d, v - d, b - d);
    }

    //! Subtraction.
    logreg_opt_vector& operator-=(value_type d) {
      f -= d;
      v -= d;
      b -= d;
      return *this;
    }

    //! Multiplication by a scalar value.
    logreg_opt_vector operator*(value_type d) const {
      return logreg_opt_vector(f * d, v * d, b * d);
    }

    //! Multiplication by a scalar value.
    logreg_opt_vector& operator*=(value_type d) {
      f *= d;
      v *= d;
      b *= d;
      return *this;
    }

    //! Division by a scalar value.
    logreg_opt_vector operator/(value_type d) const {
      assert(d != 0);
      return logreg_opt_vector(f / d, v / d, b / d);
    }

    //! Division by a scalar value.
    logreg_opt_vector& operator/=(value_type d) {
      assert(d != 0);
      f /= d;
      v /= d;
      b /= d;
      return *this;
    }

    //! Inner product with a value of the same size.
    value_type dot(const logreg_opt_vector& other) const {
      return (sill::dot(f, other.f)
              + sill::dot(v, other.v)
              + sill::dot(b, other.b));
    }

    //! Element-wise multiplication with another value of the same size.
    logreg_opt_vector& elem_mult(const logreg_opt_vector& other) {
      f %= other.f;
      v %= other.v;
      b %= other.b;
      return *this;
    }

    //! Element-wise reciprocal (i.e., change v to 1/v).
    logreg_opt_vector& reciprocal() {
      for (size_t i = 0; i < f.n_rows; ++i) {
        for (size_t j = 0; j < f.n_cols; ++j) {
          value_type& val = f(i,j);
          assert(val != 0);
          val = 1. / val;
        }
      }
      for (size_t i = 0; i < v.n_rows; ++i) {
        for (size_t j = 0; j < v.n_cols; ++j) {
          value_type& val = v(i,j);
          assert(val != 0);
          val = 1. / val;
        }
      }
      for (size_t i = 0; i < b.size(); ++i) {
        value_type& val = b[i];
        assert(val != 0);
        val = 1. / val;
      }
      return *this;
    }

    //! Returns the L1 norm.
    value_type L1norm() const {
      value_type l1val = 0;
      for (size_t i = 0; i < f.size(); ++i)
        l1val += fabs(f[i]);
      for (size_t i = 0; i < v.size(); ++i)
        l1val += fabs(v[i]);
      foreach(value_type val, b)
        l1val += fabs(val);
      return l1val;
    }

    //! Returns the L2 norm.
    value_type L2norm() const {
      return sqrt(dot(*this));
    }

    //! Returns a struct of the same size but with values replaced by their
    //! signs (-1 for negative, 0 for 0, 1 for positive).
    logreg_opt_vector sign() const {
      logreg_opt_vector ov(*this);
      for (size_t i = 0; i < f.size(); ++i)
        ov.f[i] = (f[i] > 0 ? 1 : (f[i] == 0 ? 0 : -1) );
      for (size_t i = 0; i < v.size(); ++i)
        ov.v[i] = (v[i] > 0 ? 1 : (v[i] == 0 ? 0 : -1) );
      foreach(value_type& val, ov.b)
        val = (val > 0 ? 1 : (val == 0 ? 0 : -1) );
      return ov;
    }

    //! Sets all values to 0.
    void zeros() {
      f.zeros();
      v.zeros();
      b.zeros();
    }

    //! Print info about this vector (for debugging).
    void print_info(std::ostream& out) const {
      out << "f.size: [" << f.n_rows << ", " << f.n_cols << "], "
          << "v.size: [" << v.n_rows << ", " << v.n_cols << "], "
          << "b.size: " << b.size() << "\n";
    }

    //! Print info about extrema in this vector (for debugging).
    void print_extrema_info(std::ostream& out) const {
      for (size_t i = 0; i < f.n_rows; ++i) {
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

  //! y += a * x
  template <>
  void ov_axpy<logreg_opt_vector<double> >(double a,
                                           const logreg_opt_vector<double>& x,
                                           logreg_opt_vector<double>& y);

  //! y += a * x
  template <>
  void ov_axpy<logreg_opt_vector<float> >(double a,
                                          const logreg_opt_vector<float>& x,
                                          logreg_opt_vector<float>& y);

}  // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_LOGREG_OPT_VECTOR_HPP

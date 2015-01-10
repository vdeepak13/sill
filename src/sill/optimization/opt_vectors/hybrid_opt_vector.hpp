#ifndef SILL_HYBRID_OPT_VECTOR_HPP
#define SILL_HYBRID_OPT_VECTOR_HPP

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Optimization vector defined as a list of other optimization vectors.
   *
   * \tparam Vec Type of optimization vector in the list.
   * \see hybrid_crf_factor
   */
  template <typename Vec>
  class hybrid_opt_vector {
  public:
    //! The type of values stored in this vector
    typedef typename Vec::value_type value_type;

    // Constructors
    //==========================================================================

    //! Default constructor; creates an optimization vector with 0 components.
    hybrid_opt_vector() { }

    //! Creates a vector initialized to n copies of the given component vector
    explicit hybrid_opt_vector(size_t n, const Vec& vec = Vec())
      : data(n, vec) { }

    // Accessors
    //==========================================================================

    //! Returns the number of components of this vector
    size_t size() const {
      return data.size();
    }

    //! Returns true if the vector has 0 components
    bool empty() const {
      return data.empty();
    }

    //! Returns i-th component of this vector
    const Vec& operator[](size_t i) const {
      return data[i];
    }

    //! Returns the i-th component of this vector
    Vec& operator[](size_t i) {
      return data[i];
    }

    //! Assigns n copies of the given component vector to this vector
    void assign(size_t n, const Vec& vec) {
      data.assign(n, vec);
    }

    // Vector operations
    //========================================================================

    //! Adds another vector to this one
    hybrid_opt_vector& operator+=(const hybrid_opt_vector& other) {
      check_compatible(other);
      for (size_t i = 0; i < data.size(); ++i) {
        data[i] += other.data[i];
      }
      return *this;
    }

    //! Subtracts another vector from this one
    hybrid_opt_vector& operator-=(const hybrid_opt_vector& other) {
      check_compatible(other);
      for (size_t i = 0; i < data.size(); ++i) {
        data[i] -= other.data[i];
      }
      return *this;
    }

    //! Element-wise multiplication by another vector
    hybrid_opt_vector& operator%=(const hybrid_opt_vector& other) {
      check_compatible(other);
      for (size_t i = 0; i < data.size(); ++i) {
        data[i] %= other.data[i];
      }
      return *this;
    }

    //! Multiplication by a scalar value.
    hybrid_opt_vector& operator*=(value_type x) {
      for (size_t i = 0; i < data.size(); ++i) {
        data[i] *= x;
      }
      return *this;
    }

    //! Element-wise division by another vector
    hybrid_opt_vector& operator/=(const hybrid_opt_vector& other) {
      check_compatible(other);
      for (size_t i = 0; i < data.size(); ++i) {
        data[i] /= other.data[i];
      }
      return *this;
    }

    //! Division by a scalar value.
    hybrid_opt_vector& operator/=(value_type x) {
      for (size_t i = 0; i < data.size(); ++i) {
        data[i] /= x;
      }
      return *this;
    }

    //! Returns a struct of the same size but with values replaced by their
    //! signs (-1 for negative, 0 for 0, 1 for positive).
    hybrid_opt_vector sign() const {
      hybrid_opt_vector result(size());
      for (size_t i = 0; i < result.size(); ++i) {
        result.data[i] = data[i].sign();
      }
      return result;
    }

    //! Sets all values to 0.
    void zeros() {
      for (size_t i = 0; i < data.size(); ++i) {
        data[i].zeros();
      }
    }

    //! Print info about this vector (for debugging).
    void print_info(std::ostream& out) const {
      if (empty()) {
        out << "hybrid_opt_vector(0)";
      } else {
        out << "hybrid_opt_vector(" << size() << ", ";
        print_info(out);
        out << ")";
      }
    }

    // Private data
    //==========================================================================
  private:
    std::vector<Vec> data;

  }; // class hybrid_opt_vector

  //! Outputs a vector to a stream
  template <typename Vec>
  std::ostream& operator<<(std::ostream& out, const hybrid_opt_vector<Vec>& x) {
    for (size_t i = 0; i < x.size(); ++i) {
      out << i << ": " << x[i];
    }
    return out;
  }

  //! Returns true if the two vectors are equal
  template <typename Vec>
  bool operator==(const hybrid_opt_vector<Vec>& x,
                  const hybrid_opt_vector<Vec>& y) {
    if (x.size() != y.size()) { return false; }
    for (size_t i = 0; i < x.size(); ++i) {
      if (x[i] != y[i]) { return false; }
    }
    return true;
  }

  //! Returns false if the two vectors are not equal
  template <typename Vec>
  bool operator!=(const hybrid_opt_vector<Vec>& x,
                  const hybrid_opt_vector<Vec>& y) {
    return !(x == y);
  }

  //! Inner product of two vectors
  template <typename Vec>
  typename Vec::value_type
  dot(const hybrid_opt_vector<Vec>& x, const hybrid_opt_vector<Vec>& y) {
    x.check_compatible(y);
    typename Vec::value_type sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
      sum += dot(x[i], y[i]);
    }
    return sum;
  }

  // todo: axpy

  //! Returns the L1 norm of a vector
  template <typename Vec>
  typename Vec::value_type norm_1(const hybrid_opt_vector<Vec>& x) {
    typename Vec::value_type sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
      sum += norm_1(x[i]);
    }
    return sum;
  }

  //! Returns the L2 norm of a vector
  template <typename Vec>
  typename Vec::value_type norm_2(const hybrid_opt_vector<Vec>& x) {
    return std::sqrt(dot(x, x));
  }

}; // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_HYBRID_OPT_VECTOR_HPP

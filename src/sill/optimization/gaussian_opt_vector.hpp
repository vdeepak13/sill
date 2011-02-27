#ifndef SILL_GAUSSIAN_OPT_VECTOR_HPP
#define SILL_GAUSSIAN_OPT_VECTOR_HPP

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Type which parametrizes Gaussian factors, usable for optimization and
   * learning.
   * This has: A (n x n matrix), b (n vector), C (n x m)
   * @see gaussian_crf_factor
   * @todo Generalize this to support arbitrary matrix-vector mixes.
   *
   * \ingroup optimization_classes
   */
  struct gaussian_opt_vector {

    // Types and data
    //------------------------------------------------------------------------

    struct size_type {
      //! n = |Y|
      size_t n;

      //! m = |X|
      size_t m;

      size_type(size_t n, size_t m) : n(n), m(m) { }

      bool operator==(const size_type& other) const {
        return ((n == other.n) && (m == other.m));
      }

      bool operator!=(const size_type& other) const {
        return (!operator==(other));
      }
    }; // struct size_type

    //! Size n x n
    mat A;

    //! Size n
    vec b;

    //! Size n x m
    mat C;

    // Constructors and destructors
    //------------------------------------------------------------------------

    gaussian_opt_vector() { }

    gaussian_opt_vector(size_type s, double default_val)
      : A(s.n, s.n, default_val), b(s.n, default_val),
        C(s.n, s.m, default_val) { }

    gaussian_opt_vector(const mat& A, const vec& b, const mat& C)
      : A(A), b(b), C(C) {
      if (!valid_size())
        throw std::invalid_argument
          (std::string("gaussian_opt_vector constructor:") +
           " dimensions do not match each other.");
    }

    //! Serialize members
    void save(oarchive & ar) const {
      ar << A << b << C;
    }

    //! Deserialize members
    void load(iarchive & ar) {
      ar >> A >> b >> C;
    }

    // Getters and non-math setters
    //------------------------------------------------------------------------

    //! Returns true iff this instance equals the other.
    bool operator==(const gaussian_opt_vector& other) const {
      if ((A != other.A) || (b != other.b) || (C != other.C))
        return false;
      return true;
    }

    //! Returns false iff this instance equals the other.
    bool operator!=(const gaussian_opt_vector& other) const {
      return !operator==(other);
    }

    //! Returns the dimensions of this data structure.
    size_type size() const {
      return size_type(C.size1(), C.size2());
    }

    //! Returns true iff the sizes of A, b, C match each other.
    bool valid_size() const {
      if ((A.size1() != A.size2()) || (A.size1() != b.size()) ||
          (A.size1() != C.size1()))
        return false;
      return true;
    }

    //! Resize the data.
    void resize(const size_type& newsize) {
      A.resize(newsize.n, newsize.n);
      b.resize(newsize.n);
      C.resize(newsize.n, newsize.m);
    }

    // Math operations
    //------------------------------------------------------------------------

    //! Sets all elements to this value.
    gaussian_opt_vector& operator=(double d) {
      A = d;
      b = d;
      C = d;
      return *this;
    }

    //! Addition.
    gaussian_opt_vector operator+(const gaussian_opt_vector& other) const {
      gaussian_opt_vector tmp(*this);
      tmp += other;
      return tmp;
    }

    //! Addition.
    gaussian_opt_vector& operator+=(const gaussian_opt_vector& other) {
      A += other.A;
      b += other.b;
      C += other.C;
      return *this;
    }

    //! Subtraction.
    gaussian_opt_vector operator-(const gaussian_opt_vector& other) const {
      gaussian_opt_vector tmp(*this);
      tmp -= other;
      return tmp;
    }

    //! Subtraction.
    gaussian_opt_vector& operator-=(const gaussian_opt_vector& other) {
      A -= other.A;
      b -= other.b;
      C -= other.C;
      return *this;
    }

    //! Multiplication by a scalar value.
    gaussian_opt_vector operator*(double d) const {
      gaussian_opt_vector tmp(*this);
      tmp *= d;
      return tmp;
    }

    //! Multiplication by a scalar value.
    gaussian_opt_vector& operator*=(double d) {
      A *= d;
      b *= d;
      C *= d;
      return *this;
    }

    //! Division by a scalar value.
    gaussian_opt_vector operator/(double d) const {
      gaussian_opt_vector tmp(*this);
      tmp /= d;
      return tmp;
    }

    //! Division by a scalar value.
    gaussian_opt_vector& operator/=(double d) {
      A /= d;
      b /= d;
      C /= d;
      return *this;
    }

    //! Inner product with a value of the same size.
    double inner_prod(const gaussian_opt_vector& other) const {
      return (elem_mult_sum(A, other.A)
              + sill::inner_prod(b, other.b)
              + elem_mult_sum(C, other.C));
    }

    //! Element-wise multiplication with another value of the same size.
    gaussian_opt_vector& elem_mult(const gaussian_opt_vector& other) {
      elem_mult_inplace(other.A, A);
      elem_mult_inplace(other.b, b);
      elem_mult_inplace(other.C, C);
      return *this;
    }

    //! Element-wise reciprocal (i.e., change v to 1/v).
    gaussian_opt_vector& reciprocal() {
      for (size_t i(0); i < A.size1(); ++i) {
        for (size_t j(0); j < A.size2(); ++j) {
          double& val = A(i,j);
          assert(val != 0);
          val = 1. / val;
        }
      }
      for (size_t i(0); i < b.size(); ++i) {
        double& val = b(i);
        assert(val != 0);
        val = 1. / val;
      }
      for (size_t i(0); i < C.size1(); ++i) {
        for (size_t j(0); j < C.size2(); ++j) {
          double& val = C(i,j);
          assert(val != 0);
          val = 1. / val;
        }
      }
      return *this;
    }

    //! Returns the L1 norm.
    //! WARNING: This should not be used for regularization since this factor
    //!          type supports specialized types of regularization.
    double L1norm() const {
      double l1val(0.);
      for (size_t i(0); i < A.size(); ++i)
        l1val += fabs(A(i));
      foreach(double val, b)
        l1val += fabs(val);
      for (size_t i(0); i < C.size(); ++i)
        l1val += fabs(C(i));
      return l1val;
    }

    //! Returns the L2 norm.
    //! WARNING: This should not be used for regularization since this factor
    //!          type supports specialized types of regularization.
    double L2norm() const {
      return sqrt(inner_prod(*this));
    }

    //! Returns a struct of the same size but with values replaced by their
    //! signs (-1 for negative, 0 for 0, 1 for positive).
    gaussian_opt_vector sign() const {
      gaussian_opt_vector ov(*this);
      for (size_t i(0); i < A.size(); ++i)
        ov.A(i) = (A(i) > 0 ? 1 : (A(i) == 0 ? 0 : -1) );
      foreach(double& val, ov.b)
        val = (val > 0 ? 1 : (val == 0 ? 0 : -1) );
      for (size_t i(0); i < C.size(); ++i)
        ov.C(i) = (C(i) > 0 ? 1 : (C(i) == 0 ? 0 : -1) );
      return ov;
    }

    /**
     * "Zeros" this vector by setting b, C to 0 and setting A to be a
     * diagonal matrix (identity by default).
     * @param zero_A  For this to be a "zero" factor which exerts little
     *                influence on the model, A needs to be diagonal with
     *                very small entries.  This method sets A's diagonal to
     *                this value.
     *                (default = 1)
     */
    void zeros(double zero_A = 1.) {
      A = zero_A * identity(A.size1());
      b = 0.;
      C = 0.;
    }

    void print(std::ostream& out) const {
      out << "A:\n" << A << "\n"
          << "b:\n" << b << "\n"
          << "C:\n" << C << "\n";
    }

    //! Print info about this vector (for debugging).
    void print_info(std::ostream& out) const {
      out << "A.size: [" << A.size1() << ", " << A.size2() << "], "
          << "b.size: " << b.size() << ", "
          << "C.size: [" << C.size1() << ", " << C.size2() << "]\n";
    }

  }; // struct gaussian_opt_vector

}  // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_GAUSSIAN_OPT_VECTOR_HPP

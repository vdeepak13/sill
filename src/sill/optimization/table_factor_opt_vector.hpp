#ifndef SILL_TABLE_FACTOR_OPT_VECTOR_HPP
#define SILL_TABLE_FACTOR_OPT_VECTOR_HPP

#include <sill/factor/table_factor.hpp>
#include <sill/functional/reciprocal.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Struct which permits a table_factor to be used as an OptimizationVector.
   *
   * Warning: This uses the ops built into table_factor, which do not quite
   *  fit the OptimizationVector assumptions.  For example, OptimizationVector
   *  implicitly assumes that vectors must be of the same size to be added
   *  together, but table_factor does not assume that.  (The addition of two
   *  table_factor objects creates a table_factor whose arguments are the union
   *  of the former two.)
   *
   * @see OptimizationVector
   *
   * @author Joseph Bradley
   *
   * \ingroup optimization_classes
   */
  struct table_factor_opt_vector {

    // Types and data
    //--------------------------------------------------------------------------

    typedef finite_var_vector size_type;

    table_factor f;

    // Constructors and destructors
    //--------------------------------------------------------------------------

    //! Constructor for an empty factor.
    table_factor_opt_vector() { }

    //! Constructor from a table_factor.
    explicit table_factor_opt_vector(const table_factor& f)
      : f(f) { }

    table_factor_opt_vector(const size_type& s, double default_val)
      : f(s, default_val) { }

    table_factor_opt_vector(const finite_domain& vars, double default_val)
      : f(vars, default_val) { }

    //! Serialize members
    void save(oarchive & ar) const {
      ar << f;
    }

    //! Deserialize members
    void load(iarchive & ar) {
      ar >> f;
    }

    // Getters and non-math setters
    //--------------------------------------------------------------------------

    //! Returns true iff this instance equals the other.
    bool operator==(const table_factor_opt_vector& other) const {
      return (f == other.f);
    }

    //! Returns false iff this instance equals the other.
    bool operator!=(const table_factor_opt_vector& other) const {
      return !operator==(other);
    }

    //! Returns the dimensions of this data structure.
    size_type size() const {
      return f.arg_list();
    }

    //! Resize the data.
    void resize(const size_type& newsize) {
      f = table_factor(newsize);
    }

    // Math operations
    //--------------------------------------------------------------------------

    //! Sets all elements to this value.
    table_factor_opt_vector& operator=(double d) {
      f = d;
      return *this;
    }

    //! Addition.
    table_factor_opt_vector
    operator+(const table_factor_opt_vector& other) const {
      return table_factor_opt_vector(f + other.f);
    }

    //! Addition.
    table_factor_opt_vector&
    operator+=(const table_factor_opt_vector& other) {
      f += other.f;
      return *this;
    }

    //! Addition.
    table_factor_opt_vector&
    operator+=(double d) {
      f += d;
      return *this;
    }

    //! Subtraction.
    table_factor_opt_vector
    operator-(const table_factor_opt_vector& other) const {
      return table_factor_opt_vector(f - other.f);
    }

    //! Subtraction.
    table_factor_opt_vector&
    operator-=(const table_factor_opt_vector& other) {
      f -= other.f;
      return *this;
    }

    //! Subtraction.
    table_factor_opt_vector&
    operator-=(double d) {
      f -= d;
      return *this;
    }

    //! Multiplication by a scalar value.
    table_factor_opt_vector operator*(double d) const {
      return table_factor_opt_vector(f * d);
    }

    //! Multiplication by a scalar value.
    table_factor_opt_vector& operator*=(double d) {
      f *= d;
      return *this;
    }

    //! Division by a scalar value.
    table_factor_opt_vector operator/(double d) const {
      assert(d != 0);
      return table_factor_opt_vector(f * (1./d));
    }

    //! Division by a scalar value.
    table_factor_opt_vector& operator/=(double d) {
      assert(d != 0);
      f *= (1./d);
      return *this;
    }

    //! Inner product with a value of the same size.
    double dot(const table_factor_opt_vector& other) const {
      return table_factor::combine_collapse(f, other.f,
                                            std::multiplies<double>(),
                                            std::plus<double>(), 0.0);
    }

    //! Element-wise multiplication with another value of the same size.
    table_factor_opt_vector& elem_mult(const table_factor_opt_vector& other) {
      f *= other.f;
      return *this;
    }

    //! Element-wise reciprocal (i.e., change v to 1/v).
    table_factor_opt_vector& reciprocal() {
      f.update(reciprocal_functor<double>());
      return *this;
    }

    //! Returns the L1 norm.
    double L1norm() const {
      double val(0);
      foreach(double d, f.values())
        val += fabs(d);
      return val;
    }

    //! Returns the L2 norm.
    double L2norm() const {
      return sqrt(dot(*this));
    }

    //! Returns a struct of the same size but with values replaced by their
    //! signs (-1 for negative, 0 for 0, 1 for positive).
    table_factor_opt_vector sign() const {
      table_factor_opt_vector s;
      s.f.apply(update<double, sign_functor<double> >(sign_functor<double>()));
      return s;
    }

    /**
     * Sets all values to 0.
     */
    void zeros() {
      this->operator=(0.);
    }

    //! Element-wise square root.
    void elem_square_root() {
      f.update(square_root<double>());
    }

    //! Element-wise k^th root. (I.e., raise to the (1/k)^th power.)
    void elem_kth_root(double k) {
      assert(k > 0);
      f.update(kth_root<double>(k));
    }

    //! Print info about this vector (for debugging).
    void print_info(std::ostream& out) const {
      out << "PRINT_INFO TO BE IMPLEMENTED\n";
    }

  }; // struct table_factor_opt_vector

  std::ostream& operator<<(std::ostream& out,
                           const table_factor_opt_vector& f);

}  // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_TABLE_FACTOR_OPT_VECTOR_HPP

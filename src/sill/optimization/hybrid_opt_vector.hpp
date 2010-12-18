#ifndef SILL_HYBRID_OPT_VECTOR_HPP
#define SILL_HYBRID_OPT_VECTOR_HPP

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Optimization vector defined as a list of other optimization vectors.
   *
   * @tparam SubOptVector  Type of optimization vector in the list.
   *
   * @see hybrid_crf_factor
   */
  template <typename SubOptVector>
  class hybrid_opt_vector {

    // Public types
    //==========================================================================
  public:

    typedef std::vector<typename SubOptVector::size_type> size_type;

    // Constructors and destructor
    //==========================================================================

    //! Default constructor.
    hybrid_opt_vector() : own_data_(true) { }

    //! Constructor from sub-optimization vectors.
    //! This does NOT own its data.
    hybrid_opt_vector
    (const std::vector<SubOptVector*>& sub_ov_ptrs)
      : sub_ov_ptrs(sub_ov_ptrs), own_data_(false) { }

    //! Constructor from size_type.
    //! This owns its data.
    hybrid_opt_vector(size_type s, double default_val)
      : sub_ov_ptrs(s.size(), NULL), own_data_(true) {
      for (size_t i(0); i < s.size(); ++i)
        sub_ov_ptrs[i] = new SubOptVector(s[i], default_val);
    }

    //! Copy constructor.  The copy owns its data.
    hybrid_opt_vector(const hybrid_opt_vector& other)
      : sub_ov_ptrs(other.sub_ov_ptrs.size(), NULL), own_data_(true) {
      for (size_t i(0); i < other.sub_ov_ptrs.size(); ++i)
        sub_ov_ptrs[i] = new SubOptVector(other.sub_ov_ptrs[i]);
    }

    //! Assignment operator.  The copy owns its data.
    hybrid_opt_vector& operator=(const hybrid_opt_vector& other) {
      if (own_data_) {
        foreach(SubOptVector* subptr, sub_ov_ptrs)
          delete subptr;
      }
      sub_ov_ptrs.resize(other.sub_ov_ptrs.size());
      own_data_ = true;
      for (size_t i(0); i < other.sub_ov_ptrs.size(); ++i)
        sub_ov_ptrs[i] = new SubOptVector(*(other.sub_ov_ptrs[i]));
      return *this;
    }

    ~hybrid_opt_vector() {
      if (own_data_) {
        foreach(SubOptVector* subptr, sub_ov_ptrs)
          delete subptr;
      }
    }

    // Getters and non-math setters
    //==========================================================================

    //! Returns true iff this instance equals the other.
    bool operator==(const hybrid_opt_vector& other) const {
      if (sub_ov_ptrs.size() != other.sub_ov_ptrs.size())
        return false;
      for (size_t i(0); i < sub_ov_ptrs.size(); ++i) {
        if (*(sub_ov_ptrs[i]) != *(other.sub_ov_ptrs[i]))
          return false;
      }
      return true;
    }

    //! Returns false iff this instance equals the other.
    bool operator!=(const hybrid_opt_vector& other) const {
      return !operator==(other);
    }

    //! Returns the dimensions of this data structure.
    size_type size() const {
      size_type s(sub_ov_ptrs.size());
      for (size_t i(0); i < sub_ov_ptrs.size(); ++i)
        s[i] = sub_ov_ptrs[i].size();
      return s;
    }

    //! Resize the data.
    //! This owns its data after this operation.
    void resize(const size_type& newsize) {
      if (own_data_) {
        foreach(SubOptVector* subptr, sub_ov_ptrs)
          delete subptr;
      }
      sub_ov_ptrs.resize(newsize);
      own_data_ = true;
      for (size_t i(0); i < newsize.size(); ++i)
        sub_ov_ptrs[i] = new SubOptVector(newsize[i]);
    }

    // Math operations
    //==========================================================================

    //! Sets all elements to this value.
    hybrid_opt_vector& operator=(double d) {
      foreach(SubOptVector* ovptr, sub_ov_ptrs)
        ovptr->operator=(d);
      return *this;
    }

    //! Addition.
    hybrid_opt_vector
    operator+(const hybrid_opt_vector& other) const {
      assert(sub_ov_ptrs.size() == other.sub_ov_ptrs.size());
      hybrid_opt_vector tmp(*this);
      for (size_t i(0); i < tmp.sub_ov_ptrs.size(); ++i)
        tmp.sub_ov_ptrs[i]->operator+=(other.sub_ov_ptrs[i]);
      return tmp;
    }

    //! Addition.
    hybrid_opt_vector&
    operator+=(const hybrid_opt_vector& other) {
      assert(sub_ov_ptrs.size() == other.sub_ov_ptrs.size());
      for (size_t i(0); i < sub_ov_ptrs.size(); ++i)
        sub_ov_ptrs[i]->operator+=(other.sub_ov_ptrs[i]);
      return *this;
    }

    //! Addition.
    hybrid_opt_vector&
    operator+=(double d) {
      for (size_t i(0); i < sub_ov_ptrs.size(); ++i)
        sub_ov_ptrs[i]->operator+=(d);
      return *this;
    }

    //! Subtraction.
    hybrid_opt_vector
    operator-(const hybrid_opt_vector& other) const {
      assert(sub_ov_ptrs.size() == other.sub_ov_ptrs.size());
      hybrid_opt_vector tmp(*this);
      for (size_t i(0); i < tmp.sub_ov_ptrs.size(); ++i)
        tmp.sub_ov_ptrs[i]->operator-=(other.sub_ov_ptrs[i]);
      return tmp;
    }

    //! Subtraction.
    hybrid_opt_vector&
    operator-=(const hybrid_opt_vector& other) {
      assert(sub_ov_ptrs.size() == other.sub_ov_ptrs.size());
      for (size_t i(0); i < sub_ov_ptrs.size(); ++i)
        sub_ov_ptrs[i]->operator-=(other.sub_ov_ptrs[i]);
      return *this;
    }

    //! Subtraction.
    hybrid_opt_vector&
    operator-=(double d) {
      for (size_t i(0); i < sub_ov_ptrs.size(); ++i)
        sub_ov_ptrs[i]->operator-=(d);
      return *this;
    }

    //! Multiplication by a scalar value.
    hybrid_opt_vector operator*(double d) const {
      hybrid_opt_vector tmp(*this);
      for (size_t i(0); i < tmp.sub_ov_ptrs.size(); ++i)
        tmp.sub_ov_ptrs[i]->operator*=(d);
      return tmp;
    }

    //! Multiplication by a scalar value.
    hybrid_opt_vector& operator*=(double d) {
      for (size_t i(0); i < sub_ov_ptrs.size(); ++i)
        sub_ov_ptrs[i]->operator*=(d);
      return *this;
    }

    //! Division by a scalar value.
    hybrid_opt_vector operator/(double d) const {
      assert(d != 0);
      hybrid_opt_vector tmp(*this);
      tmp *= (1. / d);
      return tmp;
    }

    //! Division by a scalar value.
    hybrid_opt_vector& operator/=(double d) {
      assert(d != 0);
      this->operator*=(1. / d);
      return *this;
    }

    //! Inner product with a value of the same size.
    double inner_prod(const hybrid_opt_vector& other) const {
      assert(sub_ov_ptrs.size() == other.sub_ov_ptrs.size());
      double sum(0);
      for (size_t i(0); i < sub_ov_ptrs.size(); ++i)
        sum += sub_ov_ptrs[i]->inner_prod(*(other.sub_ov_ptrs[i]));
      return sum;
    }

    //! Element-wise multiplication with another value of the same size.
    hybrid_opt_vector& elem_mult(const hybrid_opt_vector& other) {
      assert(sub_ov_ptrs.size() == other.sub_ov_ptrs.size());
      for (size_t i(0); i < sub_ov_ptrs.size(); ++i)
        sub_ov_ptrs[i]->elem_mult(*(other.sub_ov_ptrs[i]));
      return *this;
    }

    //! Element-wise reciprocal (i.e., change v to 1/v).
    hybrid_opt_vector& reciprocal() {
      foreach(SubOptVector* ovptr, sub_ov_ptrs)
        ovptr->reciprocal();
      return *this;
    }

    //! Returns the L1 norm.
    double L1norm() const {
      double val(0);
      foreach(SubOptVector* ovptr, sub_ov_ptrs)
        val += ovptr->L1norm();
      return val;
    }

    //! Returns the L2 norm.
    double L2norm() const {
      return sqrt(inner_prod(*this));
    }

    //! Returns a struct of the same size but with values replaced by their
    //! signs (-1 for negative, 0 for 0, 1 for positive).
    hybrid_opt_vector sign() const {
      hybrid_opt_vector s(this->size(), 0);
      for (size_t i(0); i < sub_ov_ptrs.size(); ++i)
        s.sub_ov_ptrs->operator=(sub_ov_ptrs[i]->sign());
      return s;
    }

    /**
     * Sets all values to 0.
     */
    void zeros() {
      foreach(SubOptVector* ovptr, sub_ov_ptrs)
        ovptr->operator=(0.);
    }

    // Utilities
    //==========================================================================

    //! Const reference to sub-vector i.
    const SubOptVector& subvector(size_t i) const {
      assert(i < sub_ov_ptrs.size());
      return *(sub_ov_ptrs[i]);
    }

    //! Mutable reference to sub-vector i.
    SubOptVector& subvector(size_t i) {
      assert(i < sub_ov_ptrs.size());
      return *(sub_ov_ptrs[i]);
    }

    //! Print info about this vector (for debugging).
    void print_info(std::ostream& out) const {
      out << "PRINT_INFO TO BE IMPLEMENTED\n";
    }

    // Private data
    //==========================================================================
  private:

    std::vector<SubOptVector*> sub_ov_ptrs;

    bool own_data_;

  }; // class hybrid_opt_vector

  template <typename F>
  std::ostream& operator<<(std::ostream& out,
                           const hybrid_opt_vector<F>& f) {
    assert(false); // TO DO
    return out;
  }

};  // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_HYBRID_OPT_VECTOR_HPP

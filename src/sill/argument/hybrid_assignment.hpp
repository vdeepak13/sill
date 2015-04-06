#ifndef SILL_HYBRID_ASSIGNMENT_HPP
#define SILL_HYBRID_ASSIGNMENT_HPP

#include <sill/argument/finite_assignment.hpp>
#include <sill/argument/vector_assignment.hpp>

namespace sill {

  /**
   * An assignment over a set of finite and vector variables.
   */
  template <typename T = double>
  class hybrid_assignment
    : public finite_assignment, public vector_assignment<T> {
  public:

    //! The value_type of the underlying finite assignment.
    typedef finite_assignment::value_type             finite_value_type;

    //! The value_type of the underlying vector assignment.
    typedef typename vector_assignment<T>::value_type vector_value_type;

    // Constructors and operators
    //==========================================================================

    //! Creates an empty hybrid assignment.
    hybrid_assignment() { }

    //! Constructs an assignment with the given finite component.
    hybrid_assignment(const finite_assignment& a)
      : finite_assignment(a) { }

    //! Constructs an assignment with the given vector component.
    hybrid_assignment(const vector_assignment<T>& a)
      : vector_assignment<T>(a) { }

    //! Constructs an assignment with the given components.
    hybrid_assignment(const finite_assignment& fa,
                      const vector_assignment<T>& va)
      : finite_assignment(fa),
        vector_assignment<T>(va) { }

    //! Constructs an assignment with the contents of an initializer list.
    hybrid_assignment(std::initializer_list<finite_value_type> finit)
      : finite_assignment(finit) { }

    //! Constructs an assignment with the contents fo an initializer list.
    hybrid_assignment(std::initializer_list<vector_value_type> vinit)
      : vector_assignment<T>(vinit) { }

    //! Constructs an assignment with the contents of initializer lists.
    hybrid_assignment(std::initializer_list<finite_value_type> finit,
                      std::initializer_list<vector_value_type> vinit)
      : finite_assignment(finit),
        vector_assignment<T>(vinit) { }

    //! Assignment operator.
    hybrid_assignment& operator=(const finite_assignment& a) {
      finite() = a;
      vector().clear();
      return *this;
    }

    //! Assignment operator.
    hybrid_assignment& operator=(const vector_assignment<T>& a) {
      finite().clear();
      vector() = a;
      return *this;
    }

    //! Swaps the contents of two assignments.
    friend void swap(hybrid_assignment& a, hybrid_assignment& b) {
      using std::swap;
      swap(a.finite(), b.finite());
      swap(a.vector(), b.vector());
    }

    // Accessors
    //==========================================================================
    //! Returns the finite component of this assignment.
    finite_assignment& finite() {
      return *this;
    }

    //! Returns the finite component of this assignment.
    const finite_assignment& finite() const {
      return *this;
    }

    //! Returns the vector component of this assignment.
    vector_assignment<T>& vector() {
      return *this;
    }

    //! Returns the vector component of this assignment.
    const vector_assignment<T>& vector() const {
      return *this;
    }

    //! Returns the total number of elements in this assignment.
    size_t size() const {
      return finite().size() + vector().size();
    }

    //! Returns true if the assignment is empty.
    size_t empty() const {
      return finite().empty() && vector().empty();
    }

    //! Returns 1 if the assignment contains the given finite variable.
    size_t count(finite_variable* v) const {
      return finite().count(v);
    }

    //! Returns 1 if the assignment contains the given vector variable.
    size_t count(vector_variable* v) const {
      return vector().count(v);
    }

    //! Returns 1 if the assignmemnt contains the given variable.
    bool count(variable* v) const {
      switch (v->type()) {
      case variable::FINITE_VARIABLE:
        return finite().count(dynamic_cast<finite_variable*>(v));
      case variable::VECTOR_VARIABLE:
        return vector().count(dynamic_cast<vector_variable*>(v));
      default:
        return 0;
      }
    }

    /**
     * Returns true if two assignments have the same finite and vector
     * componets.
     */
    friend bool
    operator==(const hybrid_assignment& a, const hybrid_assignment& b) {
      return a.finite() == b.finite() && a.vector() == b.vector();
    }

    /**
     * Returns true if two assignments do not have the same finite and
     * vector components.
     */
    friend bool
    operator!=(const hybrid_assignment& a, const hybrid_assignment& b) {
      return !(a == b);
    }

    // Mutations
    //==========================================================================

    //! Removes a finite variable from the assignment.
    size_t erase(finite_variable* v) {
      return finite().erase(v);
    }

    //! Removes a vector variable from the assignment.
    size_t erase(vector_variable* v) {
      return vector().erase(v);
    }

    //! Removes a variable from the assignment.
    size_t erase(variable* v) {
      switch (v->type()) {
      case variable::FINITE_VARIABLE:
        return finite_assignment::erase(dynamic_cast<finite_variable*>(v));
      case variable::VECTOR_VARIABLE:
        return vector_assignment<T>::erase(dynamic_cast<vector_variable*>(v));
      default:
        return 0;
      }
    }

    //! Removes all values from the assignment.
    void clear() {
      finite().clear();
      vector().clear();
    }

  }; // class hybrid_assignment

  /**
   * Prints a hybrid assignment to an output stream.
   * \relates hybrid_assignment
   */
  template <typename T>
  std::ostream& operator<<(std::ostream& out, const hybrid_assignment<T>& a) {
    out << a.finite();
    out << a.vector();
    return out;
  }

} // end namespace sill

#endif

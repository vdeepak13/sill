#ifndef SILL_ASSIGNMENT_HPP
#define SILL_ASSIGNMENT_HPP

#include <sill/base/finite_assignment.hpp>
#include <sill/base/vector_assignment.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * An assignment over an arbitrary set of variables.
   */
  class assignment
    : public finite_assignment, public vector_assignment {

    // Constructors and assignment/conversion operators
    //==========================================================================
  public:
    //! Creates an empty assignment
    assignment() { }

    //! Singleton constructor
    assignment(finite_variable* var, size_t value) {
      finite_assignment::insert(std::make_pair(var, value));
    }

    //! Singleton constructor
    assignment(vector_variable* var, const vec& value) { 
      vector_assignment::insert(std::make_pair(var, value));
    }

    //! Constructs an assignment with the given finite component
    assignment(const finite_assignment& a)
      : finite_assignment(a) { }

    //! Constructs an assignment with the given vector component
    assignment(const vector_assignment& a)
      : vector_assignment(a) { }

    //! Constructs an assignment with the given components
    assignment(const finite_assignment& fa, const vector_assignment& va)
      : finite_assignment(fa), vector_assignment(va) { }

    //! Assignment operator
    assignment& operator=(const finite_assignment& a) {
      // clear the other components. since a is passed by value, &a != this
      vector_assignment::clear(); 
      finite_assignment::operator=(a);
      return *this;
    }

    //! Assignment operator
    assignment& operator=(const vector_assignment& a) {
      // clear the other components. since a is passed by value, &a != this
      finite_assignment::clear();
      vector_assignment::operator=(a);
      return *this;
    }

    //! Conversion to human-readable representation
    operator std::string() const {
      using namespace sill;
      std::ostringstream out;
      out << finite_assignment(*this) << vector_assignment(*this);
      return out.str();
    }

    // Accessors
    //==========================================================================
    //! Returns the finite portion of this assignment
    const finite_assignment& finite() const {
      return *this;
    }

    //! Returns the finite portion of this assignment
    finite_assignment& finite() {
      return *this;
    }

    //! Returns the vector portion of this assignment
    const vector_assignment& vector() const {
      return *this;
    }

    //! Returns the vector portion of this assignment
    vector_assignment& vector() {
      return *this;
    }

    //! Returns the total number of elements in this assignment
    size_t size() const {
      return finite_assignment::size() + vector_assignment::size();
    }

      //! Returns true if the assignment is empty
    size_t empty() const {
      return finite_assignment::empty() && vector_assignment::empty();
    }

    // Queries
    //==========================================================================
    using finite_assignment::count;
    using vector_assignment::count;

    //! Returns 1 if the assignmemnt contains a variable
    bool count(variable* v) const {
      switch (v->type()) {
      case variable::FINITE_VARIABLE:
        return finite_assignment::count(dynamic_cast<finite_variable*>(v));
      case variable::VECTOR_VARIABLE:
        return vector_assignment::count(dynamic_cast<vector_variable*>(v));
      default:
        return 0;
      }
    }


    //  undefined for now because vector_assignment's operator== is undefined
//     //! Equality test
//     bool operator==(const assignment& a) const {
//       return this->finite() == a.finite() && this->vector() == a.vector();
//     }

//     //! Inequality test
//     bool operator!=(const assignment& a) const {
//       return !operator==(a);
//     }

    // Mutators
    //==========================================================================
    //! Removes a variable from the assignment
    using finite_assignment::erase;
    using vector_assignment::erase;

    size_t erase(variable* v) {
      switch (v->type()) {
      case variable::FINITE_VARIABLE:
        return finite_assignment::erase(dynamic_cast<finite_variable*>(v));
      case variable::VECTOR_VARIABLE:
        return vector_assignment::erase(dynamic_cast<vector_variable*>(v));
      default:
        return 0;
      }
    }

    //! Removes all elements from the assignment
    void clear() {
      finite_assignment::clear();
      vector_assignment::clear();
    }

  }; // class assignment

  
  /**
   * Computes the union of two maps
   */
  assignment map_union(const assignment& a, const assignment& b);

  /**
   * Computes the difference between two maps
   */
  assignment map_difference(const assignment& a, const assignment& b);

  /**
   * Computes the intersection of two maps
   */
  assignment map_intersect(const assignment& a, const assignment& b);

  std::ostream& operator<<(std::ostream& out, const assignment& a);  

} // end namespace sill


#include <sill/macros_undef.hpp>

#endif

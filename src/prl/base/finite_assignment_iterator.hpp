#ifndef PRL_FINITE_ASSIGNMENT_ITERATOR_HPP
#define PRL_FINITE_ASSIGNMENT_ITERATOR_HPP

#include <iterator>
#include <prl/base/finite_assignment.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  /**
   * An iterator that iterates through all possible assignments
   * to the given vector of finite variables, wrapping over when finished.
   * The order of variables in the vector dictates the order of the
   * assignments, with the first variable being the most significant digit.
   *
   * Note that this class should not be used in performance-critical code
   * For example, when computing statistics of a discrete factor, it is
   * more appropriate to perform an operation on with the underlying table.
   * In this manner, the variable-table dimension conversion is performed
   * only once.
   *
   * \todo What is the desired semantics for an iteration over an empty set?
   * \ingroup base_types
   */
  class finite_assignment_iterator
    : public std::iterator<std::forward_iterator_tag, const finite_assignment> {

    // Private data members
    //==========================================================================
  private:
    //! The ordered vector of variables to make assignments over.
    finite_var_vector var_vec;

    //! The current assignment.
    finite_assignment a;

    //! A flag indicating whether the index has wrapped around.
    bool done;

    // Public functions
    //==========================================================================
  public:
    //! Constructor. Initializes each variable to the all-0 assignment.
    explicit
    finite_assignment_iterator(const forward_range<finite_variable*>& vars);

    //! End constructor.
    finite_assignment_iterator() : done(true) { }

    //! Prefix increment.
    finite_assignment_iterator& operator++();

    //! Postfix increment.
    finite_assignment_iterator operator++(int);

    //! Returns a const reference to the current assignment.
    const finite_assignment& operator*() const {
      return a;
    }

    //! Returns a const pointer to the current assignment.
    const finite_assignment* operator->() const {
      return &a;
    }

    //! Returns truth if the two assignments are the same.
    bool operator==(const finite_assignment_iterator& it) const;

    //! Returns truth if the two assignments are different.
    bool operator!=(const finite_assignment_iterator& it) const {
      return !(*this == it);
    }

  }; // class finite_assignment_iterator
  
  /**
   * A range defined by two finite assignment iterators.
   * This type is mapped by SWIG to an appropriate type in the target language.
   * \relates finite_assignment_iterator
   */
  typedef boost::iterator_range<finite_assignment_iterator> 
    finite_assignment_range;

  //! Returns a range over all assignments to variables in the domain.
  finite_assignment_range assignments(const finite_domain& vars);


} // namespace prl

#include <prl/macros_undef.hpp>

#endif

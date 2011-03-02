
#ifndef _SILL_VECTOR_BASE_HPP_
#define _SILL_VECTOR_BASE_HPP_

#include <sill/math/sparse_linear_algebra/linear_algebra_base.hpp>

namespace sill {

  /**
   * Vector base class
   *
   * @tparam T        Type of data element (e.g., float).
   * @tparam Index    Type of index (e.g., size_t).
   */
  template <typename T, typename Index>
  class vector_base {

    // Public types
    //==========================================================================
  public:

    typedef linear_algebra_base<T,Index> la_base;

    typedef typename la_base::value_type           value_type;
    typedef typename la_base::index_type           index_type;
    typedef typename la_base::const_iterator       const_iterator;
    typedef typename la_base::iterator             iterator;
    typedef typename la_base::const_index_iterator const_index_iterator;
    typedef typename la_base::index_iterator       index_iterator;

    // Constructors
    //==========================================================================

    //! Constructor for an empty vector.
    vector_base()
      : n_(0) { }

    //! Constructor for a vector with n elements.
    vector_base(index_type n)
      : n_(n) { }

    // Getters and setters: dimensions
    //==========================================================================

    //! Length of the vector.
    index_type size() const {
      return n_;
    }

    //! Length of the vector.
    index_type length() const {
      return size();
    }

    // Operators
    //==========================================================================

    //! Equality operator.
    bool operator==(const vector_base& other) const {
      return (n_ == other.n_);
    }

    // Protected data and methods
    //==========================================================================
  protected:

    //! Number of elements.
    index_type n_;

  }; // class vector_base

} // namespace sill

#endif // #ifndef _SILL_VECTOR_BASE_HPP_

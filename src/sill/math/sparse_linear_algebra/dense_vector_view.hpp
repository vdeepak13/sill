
#ifndef _SILL_DENSE_VECTOR_VIEW_HPP_
#define _SILL_DENSE_VECTOR_VIEW_HPP_

#include <sill/math/vector.hpp>

#include <sill/math/sparse_linear_algebra/vector_base.hpp>

namespace sill {

  /**
   * Dense vector view
   *
   * This view is immutable w.r.t. values and size.
   *
   * @tparam T        Type of data element (e.g., float).
   * @tparam Index    Type of index (e.g., size_t).
   */
  template <typename T, typename Index>
  class dense_vector_view
    : public vector_base<T,Index> {

    // Public types
    //==========================================================================
  public:

    typedef vector_base<T,Index> base;

    typedef typename base::value_type           value_type;
    typedef typename base::index_type           index_type;
    typedef typename base::const_iterator       const_iterator;
    typedef typename base::iterator             iterator;
    typedef typename base::const_index_iterator const_index_iterator;
    typedef typename base::index_iterator       index_iterator;

    // Constructors
    //==========================================================================

    //! Constructor for an empty vector.
    dense_vector_view()
      : base(0), values_(NULL) { }

    // NO DESTRUCTOR.
    // This is a light view of data; it does not own the data.

    // Getters and setters: dimensions
    //==========================================================================

    using base::length;
    using base::size;

    // Getters and setters: values
    //==========================================================================

    //! Return v[i].
    //! This is NOT bound-checked.
    const value_type& operator[](index_type i) const {
      return values_[i];
    }

    //! Return v[i].
    //! This is NOT bound-checked.
    const value_type& operator()(index_type i) const {
      return operator[](i);
    }

    //! Get a const iterator to the beginning.
    const_iterator begin() const {
      return values_;
    }

    //! Get a const iterator to the end.
    const_iterator end() const {
      return values_ + n_;
    }

    // Utilities
    //==========================================================================

    // Protected data and methods
    //==========================================================================
  protected:

    using base::n_;

    //! Pointer to values.
    const value_type* values_;

  }; // class dense_vector_view

} // namespace sill

#endif // #ifndef _SILL_DENSE_VECTOR_VIEW_HPP_

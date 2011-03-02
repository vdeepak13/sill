
#ifndef _SILL_SPARSE_VECTOR_VIEW_HPP_
#define _SILL_SPARSE_VECTOR_VIEW_HPP_

#include <sill/math/sparse_linear_algebra/vector_base.hpp>
#include <sill/math/sparse_linear_algebra/dense_vector_view.hpp>

namespace sill {

  // Forward declarations
  template <typename T, typename Index> class dense_vector_view;

  /**
   * Sparse vector view
   *
   * This view is immutable w.r.t. values and size.
   *
   * This stores data in the same way as a row in the sparse matrix formats
   * Compressed Sparse Row (CSR) and Coordinate (COO).
   * This stores:
   *  - a vector of indices, indicating the location of non-zeros
   *  - a vector of values corresponding to the indices
   *
   * Design: Efficient access, slow construction.
   *  - The indices are kept sorted for efficient access.
   *  - The vector cannot be built incrementally in an efficient way.
   *  - This view type is most useful for
   *     - working with constant vectors, and
   *     - extracting views of rows or columns of matrices.
   *
   * @tparam T        Type of data element (e.g., float).
   * @tparam Index    Type of index (e.g., size_t).
   */
  template <typename T, typename Index>
  class sparse_vector_view
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
    sparse_vector_view()
      : base(0), k_(0), indices_(NULL), values_(NULL) { }

    /**
     * Constructor.
     * @param n         Length of vector.
     * @param k         Number of non-zero elements in vector.
     * @param indices_  Pointer to array of indices of non-zero elements.
     * @param values_   Pointer to array of values of non-zero elements.
     */
    sparse_vector_view(index_type n, index_type k,
                       const index_type* indices_,
                       const value_type* values_)
      : base(n), k_(k), indices_(indices_), values_(values_) { }

    // NO DESTRUCTOR.
    // This is a light view of data; it does not own the data.

    // Getters and setters: dimensions
    //==========================================================================

    using base::length;
    using base::size;

    //! Number of non-zero elements.
    index_type num_non_zeros() const {
      return k_;
    }

    // Getters and setters: values
    //==========================================================================

    /*
    //! Return v[i].
    //! This is NOT bound-checked.
    const value_type& operator[](index_type i) const {
      index_type from(0);
      index_type to(k_);
      while (from < to) {
        index_type j((from + to)/2);
        index_type ind_j(indices_[j]);
        if (ind_j == i)
          return values_[j];
        if (i < ind_j)
          to = j;
        else
          from = j + 1;
      }
      return 0;
    }
    */

    // TO DO: Add operator() and operator[] which use all threads and load
    //        coallesced chunks of indices_.

    /*
    //! Return v[i].
    //! This is NOT bound-checked.
    const value_type& operator()(index_type i) const {
      return operator[](i);
    }
    */

    //! Return the index for the i^th non-zero element.
    index_type index(index_type i) const {
      return indices_[i];
    }

    //! Return the value for the i^th non-zero element.
    value_type value(index_type i) const {
      return values_[i];
    }

    //! Get a const view of the indices.
    const dense_vector_view<index_type,index_type> indices() const {
      return dense_vector_view<index_type,index_type>(k_, indices_);
    }

    //! Get a const view of the values.
    const dense_vector_view<value_type,index_type> values() const {
      return dense_vector_view<value_type,index_type>(k_, values_);
    }

    //! Get a const iterator to the beginning of the indices.
    const_index_iterator begin_indices() const {
      return indices_;
    }

    //! Get a const iterator to the end of the indices.
    const_index_iterator end_indices() const {
      return indices_ + k_;
    }

    //! Get a const iterator to the beginning of the values.
    const_iterator begin_values() const {
      return values_;
    }

    //! Get a const iterator to the end of the values.
    const_iterator end_values() const {
      return values_ + k_;
    }

    // Utilities
    //==========================================================================

    // Protected data and methods
    //==========================================================================
  protected:

    using base::n_;

    //! Number of non-zero elements.
    index_type k_;

    //! Pointer to indices.
    const index_type* indices_;

    //! Pointer to values.
    const value_type* values_;

  }; // class sparse_vector_view

} // namespace sill

#endif // #ifndef _SILL_SPARSE_VECTOR_VIEW_HPP_

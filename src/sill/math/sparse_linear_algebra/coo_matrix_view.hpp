
#ifndef _SILL_COO_MATRIX_VIEW_HPP_
#define _SILL_COO_MATRIX_VIEW_HPP_

#include <sill/math/sparse_linear_algebra/matrix_base.hpp>

namespace sill {

  /**
   * Sparse matrix class (coordinate format): view
   *
   * This provides a view of a matrix.
   * It is immutable w.r.t. values and size.
   *
   * See coo_matrix for info on the storage format.
   *
   * @tparam T        Type of data element (e.g., float).
   * @tparam SizeType    Type of index (e.g., size_t).
   *
   * @todo Add a bit which indicates if the indices have been sorted
   *       to allow efficient access.
   */
  template <typename T, typename SizeType>
  class coo_matrix_view
    : public matrix_base<T,SizeType> {

    // Public types
    //==========================================================================
  public:

    typedef matrix_base<T,SizeType> base;

    typedef typename base::value_type           value_type;
    typedef typename base::size_type           size_type;
    typedef typename base::const_iterator       const_iterator;
    typedef typename base::iterator             iterator;
    typedef typename base::const_index_iterator const_index_iterator;
    typedef typename base::index_iterator       index_iterator;

    // Constructors
    //==========================================================================

    //! Constructor for an empty matrix.
    coo_matrix_view()
      : base(), k_(0), row_indices_(NULL), col_indices_(NULL), values_(NULL) { }

    /* // TO DO: ONLY PERMIT THIS VIA coo_matrix::view().
    //! Constructor from a host matrix.
    template <typename OtherT, typename OtherSizeType>
    coo_matrix_view(const coo_matrix<OtherT,OtherSizeType>& other)
      : base(other), k_(other.num_non_zeros()),
        row_indices_(other.row_indices_.begin()),
        col_indices_(other.col_indices_.begin()),
        values_(other.values_.begin()) { }
    */

    // NO DESTRUCTOR.
    // This should be a light view of data; it does not own the data.

    // Getters and setters: dimensions
    //==========================================================================

    using base::n_rows;
    using base::size1;
    using base::n_cols;
    using base::size2;
    using base::size;

    //! Number of non-zero elements.
    size_type num_non_zeros() const {
      return k_;
    }

    // Getters and setters: values
    //==========================================================================

    /** Get a const iterator to the beginning of the row indices. */
    const_index_iterator begin_row_indices() const {
      return row_indices_;
    }

    /** Get a const iterator to the end of the row indices. */
    const_index_iterator end_row_indices() const {
      return row_indices_ + k_;
    }

    /** Get a const iterator to the beginning of the column indices. */
    const_index_iterator begin_col_indices() const {
      return col_indices_;
    }

    /** Get a const iterator to the end of the column indices. */
    const_index_iterator end_col_indices() const {
      return col_indices_ + k_;
    }

    /** Get a const iterator to the beginning of the values. */
    const_iterator begin_values() const {
      return values_;
    }

    /** Get a const iterator to the end of the values. */
    const_iterator end_values() const {
      return values_ + k_;
    }

    // Utilities
    //==========================================================================

    // Protected data and methods
    //==========================================================================
  protected:

    using base::m_;
    using base::n_;

    //! Number of non-zero elements.
    size_type k_;

    //! Pointer to row indices.
    size_type* row_indices_;

    //! Pointer to column indices.
    size_type* col_indices_;

    //! Pointer to values.
    value_type* values_;

    /*
    //! Constructor for a matrix with the given data.
    coo_matrix_view(size_type m, size_type n,
                          size_type* col_indices_,
                          size_type* row_indices_,
                          value_type* values_)
      : base(m,n), col_indices_(col_indices_), row_indices_(row_indices_),
        values_(values_) { }
    */

  }; // class coo_matrix_view

} // namespace sill

#endif // #ifndef _SILL_COO_MATRIX_VIEW_HPP_


#ifndef _SILL_CSC_MATRIX_VIEW_HPP_
#define _SILL_CSC_MATRIX_VIEW_HPP_

#include <sill/math/sparse_linear_algebra/matrix_base.hpp>
#include <sill/math/sparse_linear_algebra/sparse_vector_view.hpp>

namespace sill {

  /**
   * Sparse matrix class (Compressed Sparse Column format): view
   *
   * This provides a view of a matrix.
   * It is immutable w.r.t. values and size.
   *
   * See csc_matrix for info on the storage format.
   *
   * @tparam T        Type of data element (e.g., float).
   * @tparam SizeType    Type of index (e.g., arma::u32).
   */
  template <typename T, typename SizeType>
  class csc_matrix_view
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
    csc_matrix_view()
      : base(), k_(0), col_offsets_(NULL), row_indices_(NULL), values_(NULL) { }

    /* // TO DO: ONLY PERMIT THIS VIA csc_matrix::view().
    //! Constructor from a host matrix.
    template <typename Allocator>
    csc_matrix_view(const csc_matrix<T,Allocator>& hmat)
      : base(hmat), k_(hmat.num_non_zeros()),
        col_offsets_(hmat.col_offsets_.begin()),
        row_indices_(hmat.row_indices_.begin()),
        values_(hmat.values_.begin()) {
    }
    */

    // NO DESTRUCTOR.
    // This is a light view of data; it does not own the data.

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

    //! Return a const view of column j of the matrix.
    sparse_vector_view<value_type,size_type> column(size_type j) const {
      if (j < n_cols()) {
        size_type co_j = col_offsets_[j];
        size_type co_jp1 = col_offsets_[j+1];
        return sparse_vector_view<value_type,size_type>
          (n_rows(), co_jp1 - co_j, row_indices_ + co_j, values_ + co_j);
      } else {
        return sparse_vector_view<value_type,size_type>();
      }
    }

    //! Look for element A(i,j).  Return <found, pointer to value>.
    std::pair<bool, const value_type*> find(size_type i, size_type j) const {
      if (i < n_rows() && j < n_cols()) {
        const size_type* row_it = row_indices_ + col_offsets_[j];
        const size_type* row_end = row_indices_ + col_offsets_[j+1];
        while (row_it < row_end) {
          if (*row_it == i)
            return std::make_pair(true, values_[row_it - row_indices_]);
          ++row_it;
        }
      }
      return std::make_pair(false, (const value_type*)NULL);
    }

    // TO DO: Add operator() and operator[] which use all threads and load
    //        coallesced chunks of indices_.

    //! Returns the offset for column i.
    //! NOTE: This does not do bound checking.
    const size_type& col_offset(size_type i) const {
      return col_offsets_[i];
    }

    //! Returns the row index for non-zero element i.
    //! NOTE: This does not do bound checking.
    size_type row_index(size_type i) const {
      return row_indices_[i];
    }

    //! Returns the value for non-zero element i.
    //! NOTE: This does not do bound checking.
    value_type value(size_type i) const {
      return values_[i];
    }

    //! Column offsets (length n+1)
    //!  col_offsets_[i] = offset in row_indices_ and values_ for column i
    //!  col_offsets_[n] = number of non-zeros
    const_index_iterator col_offsets() const {
      return col_offsets_;
    }

    //! Row indices (length k)
    const_index_iterator row_indices() const {
      return row_indices_;
    }

    //! Values (length k)
    const_iterator values() const {
      return values_;
    }

    // Utilities
    //==========================================================================

    // Protected data and methods
    //==========================================================================
  protected:

    using base::m_;
    using base::n_;

    //! Number of non-zeros.
    size_type k_;

    //! Pointer to column offsets (length n+1).
    //!  col_offsets_[i] = offset in row_indices_ and values_ for column i
    //!  col_offsets_[n] = number of non-zeros
    const size_type* col_offsets_;

    //! Pointer to row indices (length k).
    const size_type* row_indices_;

    //! Pointer to values (length k).
    const value_type* values_;

  }; // class csc_matrix_view

} // namespace sill

#endif // #ifndef _SILL_CSC_MATRIX_VIEW_HPP_

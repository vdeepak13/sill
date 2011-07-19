
#ifndef _SILL_SPARSE_VECTOR_VIEW_HPP_
#define _SILL_SPARSE_VECTOR_VIEW_HPP_

#include <sill/math/linear_algebra/dense_vector_view.hpp>
#include <sill/math/linear_algebra/sparse_vector.hpp>

namespace sill {

  // Forward declaration
  template <typename T, typename SizeType> class sparse_vector;

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
   *  - The indices are kept sorted for efficient access. (TO DO: IS THIS TRUE?)
   *  - The vector cannot be built incrementally in an efficient way.
   *  - This view type is most useful for
   *     - working with constant vectors, and
   *     - extracting views of rows or columns of matrices.
   *
   * @tparam T        Type of data element (e.g., float).
   * @tparam SizeType    Type of index (e.g., arma::u32).
   */
  template <typename T, typename SizeType>
  class sparse_vector_view
    : public vector_base<T,SizeType> {

    // Public types
    //==========================================================================
  public:

    typedef vector_base<T,SizeType> base;

    typedef typename base::value_type           value_type;
    typedef typename base::size_type           size_type;
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
    sparse_vector_view(size_type n, size_type k,
                       const size_type* indices_,
                       const value_type* values_)
      : base(n), k_(k), indices_(indices_), values_(values_) { }

    /**
     * Constructor from a sparse_vector.
     */
    sparse_vector_view(const sparse_vector<T,SizeType>& sv)
      : base(sv), k_(sv.num_non_zeros()), indices_(sv.indices().begin()),
        values_(sv.values().begin()) { }

    // NO DESTRUCTOR.
    // This is a light view of data; it does not own the data.

    // Getters and setters: dimensions
    //==========================================================================

    using base::length;
    using base::size;

    //! Number of non-zero elements.
    size_type num_non_zeros() const {
      return k_;
    }

    // Getters and setters: values
    //==========================================================================

    /*
    //! Return v[i].
    //! This is NOT bound-checked.
    const value_type& operator[](size_type i) const {
      size_type from(0);
      size_type to(k_);
      while (from < to) {
        size_type j((from + to)/2);
        size_type ind_j(indices_[j]);
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
    const value_type& operator()(size_type i) const {
      return operator[](i);
    }
    */

    //! Return the index for the i^th non-zero element.
    size_type index(size_type i) const {
      return indices_[i];
    }

    //! Return the value for the i^th non-zero element.
    value_type value(size_type i) const {
      return values_[i];
    }

    //! Get a const view of the indices.
    const dense_vector_view<size_type,size_type> indices() const {
      return dense_vector_view<size_type,size_type>(k_, indices_);
    }

    //! Get a const view of the values.
    const dense_vector_view<value_type,size_type> values() const {
      return dense_vector_view<value_type,size_type>(k_, values_);
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
    size_type k_;

    //! Pointer to indices.
    const size_type* indices_;

    //! Pointer to values.
    const value_type* values_;

  }; // class sparse_vector_view


  //! Helper function for creating views.
  template <typename T, typename SizeType>
  sparse_vector_view<T,SizeType>
  make_sparse_vector_view(const sparse_vector<T,SizeType>& v) {
    return sparse_vector_view<T,SizeType>(v);
  }

} // namespace sill

#endif // #ifndef _SILL_SPARSE_VECTOR_VIEW_HPP_

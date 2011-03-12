
#ifndef _SILL_SPARSE_VECTOR_I_HPP_
#define _SILL_SPARSE_VECTOR_I_HPP_

#include <sill/math/sparse_linear_algebra/vector_base.hpp>

namespace sill {

  // Forward declarations
  template <typename T, typename Index> class dense_vector_view;

  /**
   * Sparse vector interface.
   *
   * @see sparse_vector, sparse_vector_view
   *
   * @tparam T        Type of data element (e.g., float).
   * @tparam Index    Type of index (e.g., size_t).
   */
  template <typename T, typename Index>
  class sparse_vector_i
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
    sparse_vector_i()
      : base(0) { }

    /**
     * Constructor.
     * @param n         Length of vector.
     */
    sparse_vector_i(index_type n)
      : base(n) { }

    virtual ~sparse_vector_i() { }

    // Getters and setters: dimensions
    //==========================================================================

    using base::length;
    using base::size;

    //! Number of non-zero elements.
    virtual index_type num_non_zeros() const = 0;

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
    virtual index_type index(index_type i) const = 0;

    //! Return a mutable reference to the index for the i^th non-zero element.
    virtual index_type& index(index_type i) = 0;

    //! Return the value for the i^th non-zero element.
    virtual value_type value(index_type i) const = 0;

    //! Return a mutable reference to the value for the i^th non-zero element.
    virtual value_type& value(index_type i) = 0;

    //! Get a const view of the indices.
    virtual const dense_vector_view<index_type,index_type> indices() const = 0;

    //! Get a const view of the values.
    virtual const dense_vector_view<value_type,index_type> values() const = 0;

    //! Get a const iterator to the beginning of the indices.
    virtual const_index_iterator begin_indices() const = 0;

    //! Get a const iterator to the end of the indices.
    virtual const_index_iterator end_indices() const = 0;

    //! Get a const iterator to the beginning of the values.
    virtual const_iterator begin_values() const = 0;

    //! Get a const iterator to the end of the values.
    virtual const_iterator end_values() const = 0;

    // Utilities
    //==========================================================================

    // Protected data and methods
    //==========================================================================
  protected:

    using base::n_;

  }; // class sparse_vector_i

} // namespace sill

#endif // #ifndef _SILL_SPARSE_VECTOR_I_HPP_

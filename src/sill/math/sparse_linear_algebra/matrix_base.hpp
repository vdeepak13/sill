
#ifndef _SILL_MATRIX_BASE_HPP_
#define _SILL_MATRIX_BASE_HPP_

#include <sill/math/sparse_linear_algebra/linear_algebra_base.hpp>

namespace sill {

  /**
   * Matrix base class
   *
   * @tparam T        Type of data element (e.g., float).
   * @tparam Index    Type of index (e.g., size_t).
   */
  template <typename T, typename Index>
  class matrix_base {

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

    //! Default constructor (for an empty matrix).
    matrix_base()
      : m_(0), n_(0) { }

    //! Constructor for a matrix with m rows and n columns.
    matrix_base(index_type m, index_type n)
      : m_(m), n_(n) { }

    // Getters and setters: dimensions
    //==========================================================================

    //! Number of rows.
    index_type num_rows() const {
      return m_;
    }

    //! Number of columns.
    index_type num_cols() const {
      return n_;
    }

    //! Total number of elements (rows x columns).
    index_type size() const {
      return m_ * n_;
    }

    // Operators
    //==========================================================================

    //! Equality operator.
    bool operator==(const matrix_base& other) const {
      return (m_ == other.m_ && n_ == other.n_);
    }

    // Protected types and data
    //==========================================================================
  protected:

    //! Number of rows.
    index_type m_;

    //! Number of columns.
    index_type n_;

  }; // class matrix_base

} // namespace sill

#endif // #ifndef _SILL_MATRIX_BASE_HPP_

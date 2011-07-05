
#ifndef _SILL_MATRIX_BASE_HPP_
#define _SILL_MATRIX_BASE_HPP_

#include <sill/math/sparse_linear_algebra/linear_algebra_base.hpp>
#include <sill/serialization/serialize.hpp>

namespace sill {

  /**
   * Matrix base class
   *
   * @tparam T        Type of data element (e.g., float).
   * @tparam SizeType    Type of index (e.g., size_t).
   */
  template <typename T, typename SizeType>
  class matrix_base {

    // Public types
    //==========================================================================
  public:

    typedef linear_algebra_base<T,SizeType> la_base;

    typedef typename la_base::value_type           value_type;
    typedef typename la_base::size_type           size_type;
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
    matrix_base(size_type m, size_type n)
      : m_(m), n_(n) { }

    // Serialization
    //==========================================================================

    void save(oarchive& ar) const {
      ar << m_ << n_;
    }

    void load(iarchive& ar) {
      ar >> m_ >> n_;
    }

    // Getters and setters: dimensions
    //==========================================================================

    //! Number of rows.
//    size_type num_rows() const { return m_; }

    //! Number of rows.
    size_type n_rows() const { return m_; }

    //! Number of columns.
//    size_type num_cols() const { return n_; }

    //! Number of columns.
    size_type n_cols() const { return n_; }

    //! Total number of elements (rows x columns).
    //! NOTE: This is hard-coded to use size_t to support larger matrices.
    size_t size() const { return (size_t)m_ * (size_t)n_; }

    // Utilities
    //==========================================================================

    //! Print to the given output stream.
    virtual void print(std::ostream& out) const {
      out << "[" << m_ << " x " << n_ << " matrix]";
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
    size_type m_;

    //! Number of columns.
    size_type n_;

  }; // class matrix_base

  template <typename T, typename SizeType>
  std::ostream&
  operator<<(std::ostream& out, const matrix_base<T,SizeType>& mat) {
    mat.print(out);
    return out;
  }

} // namespace sill

#endif // #ifndef _SILL_MATRIX_BASE_HPP_

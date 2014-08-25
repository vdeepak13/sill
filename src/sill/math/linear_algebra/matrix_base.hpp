
#ifndef _SILL_MATRIX_BASE_HPP_
#define _SILL_MATRIX_BASE_HPP_

#include <sill/parsers/string_functions.hpp>
#include <sill/math/linear_algebra/linear_algebra_base.hpp>
#include <sill/serialization/serialize.hpp>

namespace sill {

  /**
   * Matrix base class
   *
   * @tparam T        Type of data element (e.g., float).
   * @tparam SizeType    Type of index (e.g., arma::u32).
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
      : n_rows(0), n_cols(0) { }

    //! Constructor for a matrix with m rows and n columns.
    matrix_base(size_type m, size_type n)
      : n_rows(m), n_cols(n) { }

    virtual ~matrix_base() { }

    // Serialization
    //==========================================================================

    void save(oarchive& ar) const {
      ar << n_rows << n_cols;
    }

    void load(iarchive& ar) {
      ar >> n_rows >> n_cols;
    }

    // Getters and setters: dimensions
    //==========================================================================

    //! Number of rows.
    size_type num_rows() const { return n_rows; }

    //! Number of columns.
    size_type num_cols() const { return n_cols; }

    //! Total number of elements (rows x columns).
    //! NOTE: This is hard-coded to use size_t to support larger matrices.
    size_t size() const { return (size_t)n_rows * (size_t)n_cols; }

    // Utilities
    //==========================================================================

    //! Print to the given output stream.
    virtual void print(std::ostream& out) const {
      out << "[" << n_rows << " x " << n_cols << " matrix]";
    }

    // Operators
    //==========================================================================

    //! Equality operator.
    bool operator==(const matrix_base& other) const {
      return (n_rows == other.n_rows && n_cols == other.n_cols);
    }

    // Public data
    //==========================================================================

    //! Number of rows.
    //! This is public to match Armadillo's interface.
    size_type n_rows;

    //! Number of columns.
    //! This is public to match Armadillo's interface.
    size_type n_cols;

  }; // class matrix_base


  template <typename T, typename SizeType>
  std::ostream&
  operator<<(std::ostream& out, const matrix_base<T,SizeType>& mat) {
    mat.print(out);
    return out;
  }

} // namespace sill

#endif // #ifndef _SILL_MATRIX_BASE_HPP_

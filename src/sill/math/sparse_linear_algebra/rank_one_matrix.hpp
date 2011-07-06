
#ifndef _SILL_RANK_ONE_MATRIX_HPP_
#define _SILL_RANK_ONE_MATRIX_HPP_

#include <sill/math/linear_algebra/armadillo.hpp>
#include <sill/math/sparse_linear_algebra/matrix_base.hpp>

namespace sill {

  /**
   * Rank-one matrix class
   *
   * This represents a matrix outer_product(x,y) as x,y,
   * where x,y are dense vectors.
   *
   * @tparam XType
   *          Type of vector x.
   *          This class takes value_type, size_type from XType.
   * @tparam YType
   *          Type of vector y.
   */
  template <typename XType, typename YType>
  class rank_one_matrix
    : public matrix_base<typename XType::value_type,typename XType::size_type>{

    // Public types
    //==========================================================================
  public:

    typedef matrix_base<typename XType::value_type,typename XType::size_type>
    base;

    typedef typename base::value_type           value_type;
    typedef typename base::size_type           size_type;
    typedef typename base::const_iterator       const_iterator;
    typedef typename base::iterator             iterator;
    typedef typename base::const_index_iterator const_index_iterator;
    typedef typename base::index_iterator       index_iterator;

    // Constructors
    //==========================================================================

    //! Constructor for an empty matrix.
    rank_one_matrix()
      : base() { }

    //! Constructor from x,y.
    rank_one_matrix(const XType& x_, const YType& y_)
      : base(x_.size(), y_.size()), x_(x_), y_(y_) { }

    // Serialization
    //==========================================================================

    void save(oarchive& ar) const {
      base::save(ar);
      ar << x_ << y_;
    }

    void load(iarchive& ar) {
      base::load(ar);
      ar >> x_ >> y_;
    }

    // Getters and setters: dimensions
    //==========================================================================

    using base::num_rows;
    using base::num_cols;
    using base::size;

    // Getters and setters: values
    //==========================================================================

    //! x (column vector in outer product)
    const XType& x() const { return x_; }

    //! y (row vector in outer product)
    const YType& y() const { return y_; }

    // Operations
    //==========================================================================

    // Utilities
    //==========================================================================

    //! Print to the given output stream.
    void print(std::ostream& out) const {
      if (this->size() == 0) {
        base::print(out);
        return;
      } else {
        out << "[" << n_rows << " x " << n_cols << " rank-1 matrix;\n"
            << " colvec = " << x_ << "\n"
            << " rowvec = " << y_ << "]\n";
      }
    } // print

    /**
     * WARNING: Only use this method if you know what you are doing!
     *
     * This deallocates any data currently in this matrix.
     * It then takes the data from the given arguments;
     * this transfers ownership of the data to this class,
     * clearing the data from the given arguments.
     * This permits piecemeal construction without unnecessary reallocation.
     */
    void reset_nocopy(XType& x, YType& y) {
      n_rows = x.size();
      n_cols = y.size();
      x_.reset_nocopy(x);
      y_.reset_nocopy(y);
    }

    using base::n_rows;
    using base::n_cols;

    // Protected data and methods
    //==========================================================================
  protected:

    //! x (column vector in outer product)
    XType x_;

    //! y (row vector in outer product)
    YType y_;

  }; // class rank_one_matrix

  //! Helper method for creating a rank_one_matrix.
  template <typename XType, typename YType>
  rank_one_matrix<XType,YType>
  make_rank_one_matrix(const XType& x, const YType& y) {
    return rank_one_matrix<XType,YType>(x,y);
  }

} // namespace sill

#endif // #ifndef _SILL_RANK_ONE_MATRIX_HPP_

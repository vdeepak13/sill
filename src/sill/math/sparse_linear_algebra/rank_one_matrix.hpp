
#ifndef _SILL_RANK_ONE_MATRIX_HPP_
#define _SILL_RANK_ONE_MATRIX_HPP_

#include <sill/math/vector.hpp>
#include <sill/math/sparse_linear_algebra/matrix_base.hpp>

namespace sill {

  /**
   * Rank-one matrix class
   *
   * This represents a matrix outer_product(x,y) as x,y,
   * where x,y are dense vectors.
   *
   * @tparam T        Type of data element (e.g., float).
   * @tparam Index    Type of index (e.g., size_t).
   */
  template <typename T, typename Index = size_t>
  class rank_one_matrix
    : public matrix_base<T,Index> {

    // Public types
    //==========================================================================
  public:

    typedef matrix_base<T,Index> base;

    typedef typename base::value_type           value_type;
    typedef typename base::index_type           index_type;
    typedef typename base::const_iterator       const_iterator;
    typedef typename base::iterator             iterator;
    typedef typename base::const_index_iterator const_index_iterator;
    typedef typename base::index_iterator       index_iterator;

    // Constructors
    //==========================================================================

    //! Constructor for an empty matrix.
    rank_one_matrix()
      : base() { }

    //! Copy constructor.
    template <typename OtherT, typename OtherIndex>
    explicit
    rank_one_matrix(const rank_one_matrix<OtherT,OtherIndex>& other)
      : base(other),
        col_offsets_(other.col_offsets()), row_indices_(other.row_indices()),
        values_(other.values()) { }

    //! Assignment operator.
    template <typename OtherT, typename OtherIndex>
    rank_one_matrix&
    operator=(const rank_one_matrix<OtherT,OtherIndex>& other) {
      m_ = other.num_rows();
      n_ = other.num_cols();
      col_offsets_ = other.col_offsets();
      row_indices_ = other.row_indices();
      values_ = other.values();
      return *this;
    }

    //! Constructor from a matrix in coordinate (COO) format.
    template <typename OtherT, typename OtherIndex>
    explicit rank_one_matrix(const coo_matrix<OtherT, OtherIndex>& other)
      : base(other.num_rows(), other.num_cols()) {
      convert_matrix(other, *this);
    }

    //! Assignment from a matrix in coordinate (COO) format.
    template <typename OtherT, typename OtherIndex>
    rank_one_matrix& operator=(const coo_matrix<OtherT, OtherIndex>& other) {
      convert_matrix(other, *this);
      return *this;
    }

    //! Constructor from a matrix in dense format.
    template <typename OtherT, typename OtherIndex>
    explicit rank_one_matrix(const matrix<OtherT>& other)
      : base(other.num_rows(), other.num_cols()) {
      convert_matrix(other, *this);
    }

    //! Assignment from a matrix in coordinate (COO) format.
    template <typename OtherT, typename OtherIndex>
    rank_one_matrix& operator=(const matrix<OtherT>& other) {
      convert_matrix(other, *this);
      return *this;
    }

    // Getters and setters: dimensions
    //==========================================================================

    using base::num_rows;
    using base::num_cols;
    using base::size;

    //! Number of non-zero elements.
    index_type num_non_zeros() const {
      return row_indices_.size();
    }

    /**
     * Resizes to m rows, n columns, and k non-zeros but does not initialize
     * data.
     * @param k  Number of non-zeros.
     *           (default = 0)
     */
    void resize(index_type m, index_type n, index_type k = 0,
                bool copy_data = false) {
      if (copy_data)
        assert(false); // TO DO
      assert(k <= m * n);
      m_ = m;
      n_ = n;
      col_offsets_.resize(k);
      row_indices_.resize(k);
      values_.resize(k);
    }

    //! Same as resize(0,0).
    void clear() {
      resize(0, 0);
    }

    // Getters and setters: values
    //==========================================================================

    //! Return a const view of column j of the matrix.
    sparse_vector_view<value_type,index_type> column(index_type j) const {
      if (j < num_cols()) {
        index_type co_j = col_offsets_[j];
        index_type co_jp1 = col_offsets_[j+1];
        return sparse_vector_view<value_type,index_type>
          (num_rows(), co_jp1 - co_j, row_indices_.begin() + co_j,
           values_.begin() + co_j);
      } else {
        return sparse_vector_view<value_type,index_type>();
      }
    }

    //! Returns element A(i,j).
    value_type operator()(index_type i, index_type j) const {
      std::pair<bool, const value_type*> found_valptr(this->find(i,j));
      if (found_valptr.first)
        return *(found_valptr.second);
      else
        return 0;
    }

    //! Look for element A(i,j).  Return <found, pointer to value>.
    std::pair<bool, const value_type*> find(index_type i, index_type j) const {
      assert(i < num_rows() && j < num_cols());
      index_type from(col_offsets_[j]);
      index_type to(col_offsets_[j+1]);
      while (from < to) {
        index_type mid((from + to) / 2);
        index_type mid_i(row_indices_[mid]);
        if (mid_i == i)
          return std::make_pair(true, &values_[mid]);
        else if (mid_i < i)
          from = mid + 1;
        else
          to = mid;
      }
      return std::make_pair(false, (const value_type*)NULL);
    }

    //! Look for element A(i,j).  Return <found, pointer to value>.
    std::pair<bool, value_type*> find(index_type i, index_type j) {
      assert(i < num_rows() && j < num_cols());
      index_type from(col_offsets_[j]);
      index_type to(col_offsets_[j+1]);
      while (from < to) {
        index_type mid((from + to) / 2);
        index_type mid_i(row_indices_[mid]);
        if (mid_i == i)
          return std::make_pair(true, &values_[mid]);
        else if (mid_i < i)
          from = mid + 1;
        else
          to = mid;
      }
      return std::make_pair(false, (value_type*)NULL);
    }

    //! Return a const view of the row indices for column j.
    const dense_vector_view<index_type,index_type>
    row_indices(index_type j) const {
      assert(j < num_cols());
      return dense_vector_view<index_type,index_type>
        (col_offsets_[j+1] - col_offsets_[j],
         row_indices_.begin() + col_offsets_[j]);
    }

    //! Return a const view of the values for column j.
    const dense_vector_view<value_type,index_type> values(index_type j) const {
      assert(j < num_cols());
      return dense_vector_view<value_type,index_type>
        (col_offsets_[j+1] - col_offsets_[j],
         values_.begin() + col_offsets_[j]);
    }

    //! Return a mutable view of the values for column j.
    dense_vector_view<value_type,index_type> values(index_type j) {
      assert(j < num_cols());
      return dense_vector_view<value_type,index_type>
        (col_offsets_[j+1] - col_offsets_[j],
         values_.begin() + col_offsets_[j]);
    }

    //! Column offsets (length n+1)
    //!  col_offsets_[i] = offset in row_indices_ and values_ for column i
    //!  col_offsets_[n] = number of non-zeros
    const vector<index_type>& col_offsets() const {
      return col_offsets_;
    }

    //! Row indices (length k)
    const vector<index_type>& row_indices() const {
      return row_indices_;
    }

    //! Values (length k)
    const vector<value_type>& values() const {
      return values_;
    }

    //! Return the maximum number of non-zeros in any column.
    index_type max_non_zeros_per_column() const {
      index_type k(0);
      for (index_type j(0); j < num_cols(); ++j) {
        k = max(k, col_offsets_[j+1] - col_offsets_[j]);
      }
      return k;
    }

    //! Return a vector of the number of non-zeros in each column.
    vector<index_type> non_zeros_per_column() const {
      if (col_offsets_.size() <= 1)
        return vector<index_type>();
      vector<index_type> sizes(col_offsets_.size() - 1);
      for (index_type j(0); j < num_cols(); ++j) {
        sizes[j] = col_offsets_[j+1] - col_offsets_[j];
      }
      return sizes;
    }

    // Operations
    //==========================================================================

    //! Returns the transpose of the matrix.
    rank_one_matrix transpose() const {
      // TO DO: Do more efficiently.
      coo_matrix<value_type, index_type> coomat(*this);
      coomat.set_transpose();
      return rank_one_matrix<value_type, index_type>(coomat);
    }

    // Utilities
    //==========================================================================

    //! Print to the given output stream.
    void print(std::ostream& out) const {
      if (this->size() == 0) {
        base::print(out);
        return;
      } else {
        out << "[" << m_ << " x " << n_ << " rank-1 matrix;\n"
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
    void reset_nocopy(vector<value_type>& x, vector<value_type>& y) {
      m_ = x_.size();
      n_ = y_.size();
      x_.reset_nocopy(x);
      y_.reset_nocopy(y);
    }

    // Protected data and methods
    //==========================================================================
  protected:

    using base::m_;
    using base::n_;

    //! x (column vector in outer product)
    vector<index_type> x_;

    //! y (row vector in outer product)
    vector<index_type> y_;

  }; // class rank_one_matrix

} // namespace sill

#endif // #ifndef _SILL_RANK_ONE_MATRIX_HPP_

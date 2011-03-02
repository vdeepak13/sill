
#ifndef _SILL_CSC_MATRIX_HPP_
#define _SILL_CSC_MATRIX_HPP_

#include <algorithm>

#include <sill/math/matrix.hpp>
#include <sill/math/vector.hpp>

#include <sill/math/sparse_linear_algebra/coo_matrix.hpp>
#include <sill/math/sparse_linear_algebra/csc_matrix_view.hpp>
#include <sill/math/sparse_linear_algebra/sparse_vector.hpp>

namespace sill {

  // Forward declarations
  template <typename T, typename Index> class csc_matrix;
  template <typename T, typename Index> class csc_matrix_view;
  template <typename T, typename Index> class coo_matrix;
  template <typename T, typename Index> class coo_matrix_view;
  template <typename T, typename Index> class sparse_vector;
  template <typename T, typename Index> class sparse_vector_view;

  namespace impl {

    //! Helper: Arrange data in standard containers.
    //!   i_val_per_column[j] = [i,value]  for each value in (i,j)
    //! @param trans  If true, do the same for coomat.transpose().
    template <typename Tfrom, typename Tto, typename Idxfrom, typename Idxto>
    void
    coo2csc_helper1
    (const coo_matrix<Tfrom, Idxfrom>& coomat,
     std::vector<std::vector<std::pair<Idxto, Tto> > >& i_val_per_column,
     bool trans = false);

    // Helper: Copy data from std containers into CSC matrix.
    //   i_val_per_column[j] = [i,value]  for each value in (i,j)
    template <typename Tfrom, typename Tto, typename Idxfrom, typename Idxto>
    void
    coo2csc_helper2
    (Idxfrom m, Idxfrom n, Idxfrom k,
     std::vector<std::vector<std::pair<Idxfrom, Tto> > >& i_val_per_column,
     csc_matrix<Tto, Idxto>& cscmat);

  } // namespace sill::impl

  template <typename Tfrom, typename Tto, typename Idxto>
  void
  convert_matrix(const matrix<Tfrom>& densemat,
                 csc_matrix<Tto, Idxto>& cscmat) {
    Idxto m = densemat.num_rows();
    Idxto n = densemat.num_cols();
    vector<Idxto> new_col_offsets(n + 1, 0);
    vector<Idxto> new_row_indices;
    vector<Tto> new_values;

    // Compute column sizes, and fill vectors of row indices and values.
    for (Idxto j(0); j < n; ++j) {
      for (Idxto i(0); i < m; ++i) {
        if (densemat(i,j) != 0) {
          ++new_col_offsets[j];
          new_row_indices.push_back(i);
          new_values.push_back(densemat(i,j));
        }
      }
    }

    // Compute offsets.
    for (Idxto i(0); i < n; ++i)
      new_col_offsets[i+1] += new_col_offsets[i];
    for (Idxto i(n); i > 0; --i)
      new_col_offsets[i] = new_col_offsets[i-1];
    new_col_offsets[0] = 0;

    // Copy data over
    cscmat.reset_nocopy(m, n, new_col_offsets, new_row_indices, new_values);
  }

  template <typename Tfrom, typename Tto, typename Idxfrom, typename Idxto>
  void
  convert_matrix(const coo_matrix<Tfrom, Idxfrom>& coomat,
                 csc_matrix<Tto, Idxto>& cscmat) {
    std::vector<std::vector<std::pair<Idxto, Tto> > > i_val_per_column;
    impl::coo2csc_helper1(coomat, i_val_per_column);
    impl::coo2csc_helper2(coomat.num_rows(), coomat.num_cols(),
                                 coomat.num_non_zeros(), i_val_per_column,
                                 cscmat);
  }

  /**
   * Sparse matrix class: Compressed Sparse Column (CSC) format
   *
   * This stores data using the Compressed Sparse Column (CSC) format,
   * which is the transpose of the Compressed Sparse Row (CSR) format.
   * See "Efficient Sparse Matrix-Vector Multiplication on CUDA" by
   * Nathan Bell and Michael Garland for details on sparse matrix formats.
   * This stores:
   *  - a vector of row indices, indicating the location of non-zeros
   *  - a pointer vector which gives, for each column, the offset in the
   *    row index vector for that column
   *  - a vector of values corresponding to the vector of row indices
   *
   * Design: Efficient access, slow construction.
   *  - The indices are kept sorted for efficient access.
   *  - The matrix cannot be built incrementally in an efficient way.
   *  - This view type is most useful for working with constant matrices.
   *
   * @tparam T        Type of data element (e.g., float).
   * @tparam Index    Type of index (e.g., size_t).
   */
  template <typename T, typename Index = size_t>
  class csc_matrix
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
    csc_matrix()
      : base() { }

    //! Copy constructor.
    template <typename OtherT, typename OtherIndex>
    explicit
    csc_matrix(const csc_matrix<OtherT,OtherIndex>& other)
      : base(other),
        col_offsets_(other.col_offsets()), row_indices_(other.row_indices()),
        values_(other.values()) { }

    //! Assignment operator.
    template <typename OtherT, typename OtherIndex>
    csc_matrix& operator=(const csc_matrix<OtherT,OtherIndex>& other) {
      m_ = other.num_rows();
      n_ = other.num_cols();
      col_offsets_ = other.col_offsets();
      row_indices_ = other.row_indices();
      values_ = other.values();
      return *this;
    }

    //! Constructor from a matrix in coordinate (COO) format.
    template <typename OtherT, typename OtherIndex>
    explicit csc_matrix(const coo_matrix<OtherT, OtherIndex>& other)
      : base(other.num_rows(), other.num_cols()) {
      convert_matrix(other, *this);
    }

    //! Assignment from a matrix in coordinate (COO) format.
    template <typename OtherT, typename OtherIndex>
    csc_matrix& operator=(const coo_matrix<OtherT, OtherIndex>& other) {
      convert_matrix(other, *this);
      return *this;
    }

    //! Constructor from a matrix in dense format.
    template <typename OtherT, typename OtherIndex>
    explicit csc_matrix(const matrix<OtherT>& other)
      : base(other.num_rows(), other.num_cols()) {
      convert_matrix(other, *this);
    }

    //! Assignment from a matrix in coordinate (COO) format.
    template <typename OtherT, typename OtherIndex>
    csc_matrix& operator=(const matrix<OtherT>& other) {
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
    csc_matrix transpose() const {
      // TO DO: Do more efficiently.
      coo_matrix<value_type, index_type> coomat(*this);
      coomat.set_transpose();
      return csc_matrix<value_type, index_type>(coomat);
    }

    /**
     * Matrix-vector element-wise apply.
     *
     * For each column j of this matrix M,
     *   For each row i in the column,
     *     Set M(i,j) = op(M(i,j), v(j)).
     *
     * @param trans  If 'n','N', use M; i.e., apply v(j) to column j of M.
     *               If 't','T','c','C', use M'; i.e., apply v(j) to row j of M.
     * @param v      Input vector.
     * @param op     Binary operator.
     */
    /*
    template <typename Op>
    void ew_apply_vector(const dense_vector<T>& v, char trans,
                         Op op) {
      switch (trans) {
      case 'n': case 'N':
        if (v.size() != num_cols()) {
          throw std::invalid_argument("csc_matrix::vector_ew_apply given vector not matching matrix dimensions");
        }
        for (index_type j(0); j < num_cols(); ++j) {
          index_type co_j = col_offsets_[j];
          index_type co_jp1 = col_offsets_[j+1];
          while (co_j < co_jp1) {
            values_[co_j] = op(values_[co_j], v[j]);
            ++co_j;
          }
        }
        break;
      case 't': case 'T': case 'c': case 'C':
        assert(false); // TO DO
        break;
      default:
        assert(false);
      }
    }
    */

    // Utilities
    //==========================================================================

    //! Print to the given output stream.
    void print(std::ostream& out) const {
      if (this->size() == 0) {
        out << "[]";
        return;
      } else {
        for (index_type j(0); j < num_cols(); ++j) {
          const dense_vector_view<index_type,index_type>
            col_j_indices(row_indices(j));
          const dense_vector_view<value_type,index_type>
            col_j_values(values(j));
          if (j == 0)
            out << "[";
          else
            out << " ";
          out << "(*," << j << "): ";
          for (index_type i(0); i < col_j_indices.size(); ++i) {
            out << col_j_indices[i] << "(" << col_j_values[i] << ")";
            if (i+1 != col_j_indices.size())
              out << ", ";
          }
          if (j + 1 != num_cols())
            out << "\n";
          else
            out << "]\n";
        }
      }
    } // print

    /**
     * WARNING: Only use this method if you know what you are doing!
     *
     * This deallocates any data currently in this matrix.
     * It then takes the data from the given arguments;
     * this transfers ownership of the data to this class,
     * clearing the data from the given arguments.
     * This permits piecemeal construction of a CSC matrix without
     * unnecessary reallocation.
     *
     * @param new_row_indices   WARNING: These MUST be sorted in increasing
     *                          order (for each column).
     */
    void reset_nocopy(index_type m, index_type n,
                      vector<index_type>& new_col_offsets,
                      vector<index_type>& new_row_indices,
                      vector<value_type>& new_values) {
      m_ = m;
      n_ = n;
      col_offsets_.reset_nocopy(new_col_offsets);
      row_indices_.reset_nocopy(new_row_indices);
      values_.reset_nocopy(new_values);
    }

    //! Number of bytes to represent this matrix.
    static index_type bytes_required(index_type n_non_zeros) {
      return sizeof(csc_matrix)
        + n_non_zeros * (2 * sizeof(index_type) + sizeof(value_type));
    }

    // Protected data and methods
    //==========================================================================
  protected:

    using base::m_;
    using base::n_;

    //! Column offsets (length n+1)
    //!  col_offsets_[i] = offset in row_indices_ and values_ for column i
    //!  col_offsets_[n] = number of non-zeros
    vector<index_type> col_offsets_;

    //! Row indices (length k)
    vector<index_type> row_indices_;

    //! Values (length k)
    vector<value_type> values_;

  }; // class csc_matrix

  template <typename T, typename Index>
  std::ostream&
  operator<<(std::ostream& out, const csc_matrix<T,Index>& mat) {
    mat.print(out);
    return out;
  }

  namespace impl {

    // Definitions of impl methods declared above.
    //==========================================================================

    template <typename Tfrom, typename Tto, typename Idxfrom, typename Idxto>
    void
    coo2csc_helper1
    (const coo_matrix<Tfrom, Idxfrom>& coomat,
     std::vector<std::vector<std::pair<Idxto, Tto> > >& i_val_per_column,
     bool trans = false) {
      typedef typename coo_matrix<Tfrom,Idxfrom>::const_index_iterator
        const_index_iterator;
      typedef typename coo_matrix<Tfrom,Idxfrom>::const_iterator const_iterator;
      const_index_iterator row_it, col_it, row_end;
      const_iterator val_it;
      if (trans) {
        i_val_per_column.resize(coomat.num_rows());
        row_it = coomat.col_indices().begin();
        col_it = coomat.row_indices().begin();
        val_it = coomat.values().begin();
        row_end = coomat.col_indices().end();
      } else {
        i_val_per_column.resize(coomat.num_cols());
        row_it = coomat.row_indices().begin();
        col_it = coomat.col_indices().begin();
        val_it = coomat.values().begin();
        row_end = coomat.row_indices().end();
      }
      while (row_it != row_end) {
        i_val_per_column[*col_it].push_back
          (std::make_pair<Idxfrom, Tto>(*row_it, *val_it));
        ++row_it;
        ++col_it;
        ++val_it;
      }
    } // coo2csc_helper1

    template <typename Tfrom, typename Tto, typename Idxfrom, typename Idxto>
    void
    coo2csc_helper2
    (Idxfrom m, Idxfrom n, Idxfrom k,
     std::vector<std::vector<std::pair<Idxfrom, Tto> > >& i_val_per_column,
     csc_matrix<Tto, Idxto>& cscmat) {

      assert(n == i_val_per_column.size());

      // Compute offsets
      vector<Idxfrom> new_col_offsets(n + 1);
      Idxfrom total(0);
      for (Idxfrom i(0); i < n; ++i) {
        Idxfrom offset(total);
        total += i_val_per_column[i].size();
        new_col_offsets[i] = offset;
      }
      new_col_offsets[n] = total;
      assert(total == k);

      // Copy data over
      vector<Idxfrom> new_row_indices(k);
      vector<Tto> new_values(k);
      total = 0;
      for (Idxfrom i(0); i < n; ++i) {
        std::vector<std::pair<Idxfrom, Tto> >& i_val_pairs =i_val_per_column[i];
        std::sort(i_val_pairs.begin(), i_val_pairs.end());
        for (Idxfrom j(0); j < i_val_pairs.size(); ++j) {
          new_row_indices[total] = i_val_pairs[j].first;
          new_values[total] = i_val_pairs[j].second;
          ++total;
        }
      }
      assert(total == k);
      cscmat.reset_nocopy(m, n, new_col_offsets, new_row_indices, new_values);
    } // coo2csc_helper2

  } // namespace impl

} // namespace sill

#endif // #ifndef _SILL_CSC_MATRIX_HPP_

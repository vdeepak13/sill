
#ifndef _SILL_COO_MATRIX_HPP_
#define _SILL_COO_MATRIX_HPP_

#include <sill/math/linear_algebra/armadillo.hpp>
#include <sill/math/sparse_linear_algebra/coo_matrix_view.hpp>
#include <sill/math/sparse_linear_algebra/csc_matrix.hpp>
#include <sill/math/sparse_linear_algebra/sparse_vector.hpp>

namespace sill {

  // Forward declarations
  template <typename T, typename SizeType> class coo_matrix_view;
  template <typename T, typename SizeType> class csc_matrix;
  template <typename T, typename SizeType> class csc_matrix_view;
  template <typename T, typename SizeType> class sparse_vector;
  template <typename T, typename SizeType> class sparse_vector_view;

  /**
   * Sparse matrix class: Coordinate (COO) format
   *
   * This stores data using the Coordinate (COO) format.
   * See "Efficient Sparse Matrix-Vector Multiplication on CUDA" by
   * Nathan Bell and Michael Garland for details on sparse matrix formats.
   * This stores:
   *  - a vector of row indices, indicating the location of non-zeros
   *  - a vector of column indices corresponding to the row indices
   *  - a vector of values corresponding to the vectors of indices
   *
   * Design: Fast construction.
   *  - The indices are NOT kept sorted for efficient access.
   *  - The matrix can be built incrementally in an efficient way.
   *
   * @tparam T        Type of data element (e.g., float).
   * @tparam SizeType    Type of index (e.g., size_t).
   */
  template <typename T, typename SizeType = size_t>
  class coo_matrix
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
    coo_matrix()
      : base(), k_(0) { }

    /**
     * Constructor for a matrix with m rows, n columns, and
     * the reserved capacity for k non-zeros.
     * @param k  (default = 0)
     */
    coo_matrix(size_type m, size_type n, size_type k = 0)
      : base(m,n), k_(k) {
      resize_data(k, false);
    }

    //! Constructor from another type.
    template <typename OtherT, typename OtherSizeType>
    explicit coo_matrix(const coo_matrix<OtherT,OtherSizeType>& other)
      : base(other.n_rows(), other.n_cols()), k_(other.num_non_zeros()),
        row_indices_(other.row_indices()), col_indices_(other.col_indices()),
        values_(other.values()) {
    }

    //! Constructor from another type.
    template <typename OtherT, typename OtherSizeType>
    explicit coo_matrix(const csc_matrix<OtherT,OtherSizeType>& other)
      : base(other.n_rows(), other.n_cols()), k_(other.num_non_zeros()),
        row_indices_(k_), col_indices_(k_), values_(k_) {
      this->operator=(other);
    }

    //! Assignment from a matrix in CSC format.
    template <typename OtherT, typename OtherSizeType>
    coo_matrix& operator=(const csc_matrix<OtherT,OtherSizeType>& other) {
      base::operator=(other);
      k_ = other.num_non_zeros();
      row_indices_.resize(k_);
      col_indices_.resize(k_);
      values_.resize(k_);
      size_type l(0);
      for (size_type j(0); j < other.n_cols(); ++j) {
        const sparse_vector_view<OtherT,OtherSizeType>
          other_col_j(other.column(j));
        for (size_type i(0); i < other_col_j.num_non_zeros(); ++i) {
          row_indices_[l] = other_col_j.index(i);
          col_indices_[l] = j;
          values_[l] = other_col_j.value(i);
          ++l;
        }
      }
      return *this;
    }

    // Serialization
    //==========================================================================

    void save(oarchive& ar) const {
      base::save(ar);
      ar << row_indices_ << col_indices_ << values_;
    }

    void load(iarchive& ar) {
      base::load(ar);
      ar >> row_indices_ >> col_indices_ >> values_;
    }

    // Getters and setters: dimensions
    //==========================================================================

    using base::n_cols;
    using base::size1;
    using base::n_rows;
    using base::size2;
    using base::size;

    /**
     * Resizes the matrix to have m rows, n columns.
     *
     * @param k  Number of non-zeros.
     *           (default = 0)
     * @param copy_data  If true, then the overlapping part of the old and new
     *                   matrices will be copied over to the new matrix.
     *                   If false, then the matrix begins with no reserved
     *                   space for non-zeros.
     *                   (default = false)
     */
    void resize(size_type m, size_type n, size_type k = 0,
                bool copy_data = false) {
      if (m == m_ && n == n_ && k == k_)
        return;
      if (k > m * n) {
        throw std::invalid_argument
          (std::string("coo_matrix<T>::resize(m,n,k,copy_data)")
           + " called with k > m*n. k = " + to_string(k) + ", m = "
           + to_string(m) + ", n = " + to_string(n));
      }
      if (copy_data) {
        assert(false); // TO DO
      } else {
        resize_data(k, false);
        m_ = m;
        n_ = n;
        k_ = k;
      }
    }

    //! Number of spaces reserved for non-zeros.
    size_type capacity() const {
      return row_indices_.size();
    }

    /**
     * Reserves space for cap non-zeros.
     * If cap > m * n, then this throws a std::invalid_argument exception.
     * If cap <= current capacity, then this does nothing.
     */
    void reserve(size_type cap) {
      if (cap <= capacity())
        return;
      if (cap > m_ * n_)
        throw std::invalid_argument
          ("coo_matrix::reserve() was given capacity > n_rows * n_cols.");
      resize_data(cap, true);
    }

    // Getters and setters: values
    //==========================================================================

    //! Return a mutable reference to A(i,j).
    //! This is NOT bound-checked.
    value_type& operator()(size_type i, size_type j) {
      std::pair<bool, value_type*> found_val(find(i,j));
      if (found_val.first) {
        return *(found_val.second);
      } else {
        if (k_ >= capacity())
          reserve(k_ + 1);
        size_type old_k(k_);
        ++k_;
        row_indices_[old_k] = i;
        col_indices_[old_k] = j;
        return values_[old_k];
      }
    }

    //! Return a mutable reference to A(i,j).
    //! This is NOT bound-checked.
    value_type operator()(size_type i, size_type j) const {
      std::pair<bool, value_type*> found_val(find(i,j));
      if (found_val.first) {
        return *(found_val.second);
      } else {
        return 0;
      }
    }

    //! Number of non-zero elements.
    size_type num_non_zeros() const {
      return k_;
    }

    //! Sets the number of non-zero elements; this is helpful if building the
    //! matrix by directly inserting elements into row_indices, col_indices,
    //! and values.
    void set_num_non_zeros(size_type new_k) {
      assert(new_k <= m_ * n_);
      if (k_ > row_indices_.size())
        resize_data(new_k, true);
      k_ = new_k;
    }

    //! i^th row index.
    size_type row_index(size_type i) const {
      assert(i < k_);
      return row_indices_[i];
    }

    //! i^th column index.
    size_type col_index(size_type i) const {
      assert(i < k_);
      return col_indices_[i];
    }

    //! i^th non-zero value.
    value_type value(size_type i) const {
      assert(i < k_);
      return values_[i];
    }

    //! Row indices.
    arma::Col<size_type>& row_indices() {
      return row_indices_;
    }

    //! Column indices.
    arma::Col<size_type>& col_indices() {
      return col_indices_;
    }

    //! Values.
    arma::Col<value_type>& values() {
      return values_;
    }

    //! Row indices.
    const arma::Col<size_type>& row_indices() const {
      return row_indices_;
    }

    //! Column indices.
    const arma::Col<size_type>& col_indices() const {
      return col_indices_;
    }

    //! Values.
    const arma::Col<value_type>& values() const {
      return values_;
    }

    //! Sets this matrix to be its own transpose.
    void set_transpose() {
      row_indices().swap(col_indices());
      std::swap(m_, n_);
    }

    //! Returns the transpose of this matrix.
    void transpose() const {
      coo_matrix m(*this);
      m.set_tranpose();
      return m;
    }

    //! Sets matrix to all zeros.
    void zeros() {
      k_ = 0;
    }

    //! Return a vector of the number of non-zeros in each column.
    arma::Col<size_type> non_zeros_per_column() const {
      arma::Col<size_type> sizes(n_cols(), 0);
      for (size_type k(0); k < num_non_zeros(); ++k)
        ++sizes[col_index(k)];
      return sizes;
    }

    // Operators
    //==========================================================================

    //! Equality operator.
////    bool operator==(const coo_matrix& other) const {

    // Utilities
    //==========================================================================

    /**
     * Permute the columns of this matrix.
     * @param perm  Permutation: perm[i] = new column index for column i
     */
    void permute_columns(const arma::Col<size_type>& perm) {
      if (perm.size() != n_cols()) {
        throw std::runtime_error
          (std::string("coo_matrix<T>::permute_columns(perm)") +
           " was given a permutation of an incorrect size.");
      }
      for (iterator it = col_indices().begin();
           it != col_indices().end();
           ++it) {
        *it = perm[*it];
      }
    }

    /**
     * Remove columns which are all-zero.
     * This shifts the indices of other columns as necessary.
     * @return vector[i] = new column index for original column i
     */
    arma::Col<size_type> remove_all_zero_cols() {
      arma::Col<size_type> col_sizes(non_zeros_per_column());
      if (col_sizes.size() == 0)
        return col_sizes;
      // TO DO: Replace the below code with a prescan call.
      size_type carry(col_sizes[0] != 0 ? 1 : 0);
      col_sizes[0] = 0;
      for (size_type k(1); k < col_sizes.size(); ++k) {
        size_type next_carry = (col_sizes[k] != 0 ? 1 : 0);
        col_sizes[k] = col_sizes[k-1] + carry;
        carry = next_carry;
      }
      permute_columns(col_sizes);
      unsafe_set_mnk(n_rows(), col_sizes.last() + carry, num_non_zeros());
      return col_sizes;
    }

    //! Print to the given output stream.
    void print(std::ostream& out) const {
      if (size() == 0) {
        out << "[]";
        return;
      } else {
        const_index_iterator row_it = row_indices().begin();
        const_index_iterator col_it = col_indices().begin();
        const_iterator value_it = values().begin();
        size_type i = 0;
        while (row_it != row_indices().end()) {
          if (i == 0)
            out << "[";
          else
            out << " ";
          out << "(" << (*row_it) << "," << (*col_it) << "): " << (*value_it);
          if (i != num_non_zeros() - 1)
            out << "\n";
          else
            out << "]\n";
          ++row_it;
          ++col_it;
          ++value_it;
          ++i;
        }
      }
    }

    /**
     * WARNING: Only use this method if you know what you are doing!
     *
     * This deallocates any data currently in this matrix.
     * It then takes the data from the given arguments;
     * this transfers ownership of the data to this class,
     * clearing the data from the given arguments.
     * This permits piecemeal construction of a COO matrix without
     * unnecessary reallocation.
     *
     * This does not currently check bounds for the given indices.
     */
    void reset_nocopy(size_type m, size_type n,
                      arma::Col<size_type>& new_row_indices,
                      arma::Col<size_type>& new_col_indices,
                      arma::Col<value_type>& new_values) {
      k_ = new_row_indices.size();
      if (m * n < k_ ||
          new_col_indices.size() != k_ ||
          new_values.size() != k_) {
        throw std::invalid_argument
          (std::string("coo_matrix<T>::reset_nocopy") +
           " was given arguments with non-matching dimensions.");
      }
      m_ = m;
      n_ = n;
      row_indices_.reset_nocopy(new_row_indices);
      col_indices_.reset_nocopy(new_col_indices);
      values_.reset_nocopy(new_values);
    }

    //! Set m,n,k by force.
    //! WARNING: Do not use this unless you know what you are doing!
    void
    unsafe_set_mnk(size_type newm, size_type newn, size_type newk) {
      m_ = newm;
      n_ = newn;
      k_ = newk;
    }

    // Protected data and methods
    //==========================================================================
  protected:

    using base::m_;
    using base::n_;

    //! Number of non-zero elements.
    size_type k_;

    arma::Col<size_type> row_indices_;

    arma::Col<size_type> col_indices_;

    arma::Col<value_type> values_;

    //! Resize all vectors, copying the data.
    void resize_data(size_type cap, bool copy_data) {
      row_indices_.resize(cap, copy_data);
      col_indices_.resize(cap, copy_data);
      values_.resize(cap, copy_data);
    }

    //! Look for element A(i,j).  Return <found, pointer to value>.
    std::pair<bool, const value_type*> find(size_type i, size_type j) const {
      // TO DO: Update this when I add the option to sort the values
      //        by index.
      for (size_type p(0); p < k_; ++p) {
        if ((row_indices_[p] == i) && (col_indices_[p] == j))
          return std::make_pair(true, &values_[p]);
      }
      return std::make_pair(false, (value_type*)NULL);
    }

    //! Look for element A(i,j).  Return <found, pointer to value>.
    std::pair<bool, value_type*> find(size_type i, size_type j) {
      // TO DO: Update this when I add the option to sort the values
      //        by index.
      for (size_type p(0); p < k_; ++p) {
        if ((row_indices_[p] == i) && (col_indices_[p] == j))
          return std::make_pair(true, &values_[p]);
      }
      return std::make_pair(false, (value_type*)NULL);
    }

  }; // class coo_matrix

}; // namespace sill

#endif // #ifndef _SILL_COO_MATRIX_HPP_

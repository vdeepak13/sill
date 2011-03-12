
#ifndef _SILL_SPARSE_VECTOR_HPP_
#define _SILL_SPARSE_VECTOR_HPP_

#include <sill/math/vector.hpp>

#include <sill/math/sparse_linear_algebra/sparse_vector_view.hpp>

namespace sill {

  /**
   * Sparse host/device vector class.
   *
   * @tparam T        Type of data element (e.g., float).
   * @tparam Index    Type of index (e.g., size_t).
   */
  template <typename T, typename Index = size_t>
  class sparse_vector
    : public sparse_vector_i<T,Index> {

    // Public types
    //==========================================================================
  public:

    typedef sparse_vector_i<T,Index> base;

    typedef typename base::value_type           value_type;
    typedef typename base::index_type           index_type;
    typedef typename base::const_iterator       const_iterator;
    typedef typename base::iterator             iterator;
    typedef typename base::const_index_iterator const_index_iterator;
    typedef typename base::index_iterator       index_iterator;

    // Constructors
    //==========================================================================

    //! Constructs an empty vector.
    sparse_vector()
      : base(0) { }

    //! Constructs a vector with n elements and k non-zero elements,
    //! but does not initialize them.
    sparse_vector(index_type n, index_type k)
      : base(n), indices_(k), values_(k) {
      assert(k <= n);
    }

    /**
     * Constructs a vector with the given non-zero elements.
     * @param n         Size of vector.
     * @param indices_  Indices of non-zeros elements.
     * @param values_   Corresponding values of non-zero elements.
     */
    template <typename IndexVecType, typename ValueVecType>
    sparse_vector(index_type n, const IndexVecType& indices_,
                  const ValueVecType& values_)
      : base(n), indices_(indices_), values_(values_) {
      assert(indices_.size() == values_.size());
      for (index_type i(0); i < indices_.size(); ++i) {
        if (indices_[i] >= n)
          throw std::invalid_argument
            ("sparse_vector constructor given non-zero index "
             + to_string(indices_[i]) + " too large for vector of size "
             + to_string(n));
      }
    }

    // Getters and setters: dimensions
    //==========================================================================

    using base::length;
    using base::size;

    //! Number of non-zero elements.
    index_type num_non_zeros() const {
      return indices_.size();
    }

    //! Resizes for n elements and k non-zeros but does not initialize the
    //! elements.
    //! NOTE: This does NOT currently retain old values.
    void resize(index_type n, index_type k) {
      this->n_ = n;
      indices_.resize(k);
      values_.resize(k);
    }

    // Getters and setters: values
    //==========================================================================

    //! Return the index for the i^th non-zero element.
    index_type index(index_type i) const {
      return indices_[i];
    }

    //! Return a mutable reference to the index for the i^th non-zero element.
    index_type& index(index_type i) {
      return indices_[i];
    }

    //! Return the value for the i^th non-zero element.
    value_type value(index_type i) const {
      return values_[i];
    }

    //! Return a mutable reference to the value for the i^th non-zero element.
    value_type& value(index_type i) {
      return values_[i];
    }

    //! Indices of non-zeros.
    const dense_vector_view<index_type,index_type> indices() const {
      return dense_vector_view<index_type,index_type>(indices_);
    }

    //! Values of non-zeros.
    const dense_vector_view<value_type,index_type> values() const {
      return dense_vector_view<value_type,index_type>(values_);
    }

    //! Get a const iterator to the beginning of the indices.
    const_index_iterator begin_indices() const {
      return indices_.begin();
    }

    //! Get a const iterator to the end of the indices.
    const_index_iterator end_indices() const {
      return indices_.end();
    }

    //! Get a const iterator to the beginning of the values.
    const_iterator begin_values() const {
      return values_.begin();
    }

    //! Get a const iterator to the end of the values.
    const_iterator end_values() const {
      return values_.end();
    }

    // Utilities
    //==========================================================================

    //! Swap this vector with another efficiently.
    //! NOTE: This swaps pointers, so any iterators will be invalidated.
    void swap(sparse_vector& other) {
      indices_.swap(other.indices_);
      values_.swap(other.values_);
    }

    //! Print to the given output stream.
    void print(std::ostream& out) const {
      out << "[n=" << size() << "; ";
      for (index_type i(0); i < num_non_zeros(); ++i)
        out << index(i) << "(" << value(i) << ") ";
      out << "]";
    }

    // Operations
    //==========================================================================

    //! Set to all zeros.
    void zeros() {
      indices_.resize(0);
      values_.resize(0);
    }

    // Protected data and methods
    //==========================================================================
  protected:

    using base::n_;

    //! Indices of non-zeros.
    vector<index_type> indices_;

    //! Values of non-zeros.
    vector<value_type> values_;

  }; // class sparse_vector

  template <typename T, typename Index>
  std::ostream&
  operator<<(std::ostream& out, const sparse_vector<T,Index>& vec) {
    vec.print(out);
    return out;
  }

}; // namespace sill

#endif // #ifndef _SILL_SPARSE_VECTOR_HPP_

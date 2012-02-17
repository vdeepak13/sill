
#ifndef _SILL_DENSE_VECTOR_VIEW_HPP_
#define _SILL_DENSE_VECTOR_VIEW_HPP_

#include <sill/math/linear_algebra/armadillo.hpp>
#include <sill/math/linear_algebra/vector_base.hpp>

namespace sill {

  /**
   * Dense vector view
   *
   * This view is immutable w.r.t. values and size.
   *
   * This class supports variable pitch, i.e., views which skip elements
   * in a periodic fashion.
   * E.g., a view of [1 2 3 4 5 6] with size 2 and pitch 3 would be [1 4].
   *
   * @tparam T        Type of data element (e.g., float).
   * @tparam SizeType    Type of index (e.g., arma::u32).
   */
  template <typename T, typename SizeType>
  class dense_vector_view
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
    dense_vector_view()
      : base(0), values_(NULL), pitch_(1) { }

    //! Constructor with data.
    dense_vector_view(size_type n, const_iterator it, size_type pitch_ = 1)
      : base(n), values_(it), pitch_(pitch_) { }

    //! Constructor from a dense vector (with pitch = 1).
    explicit dense_vector_view(const arma::Col<T>& v)
      : base(v.size()), values_(v.begin()), pitch_(1) { }

    // NO DESTRUCTOR.
    // This is a light view of data; it does not own the data.

    // Getters and setters: dimensions
    //==========================================================================

    using base::length;
    using base::size;

    //! Pitch
    size_type pitch() const { return pitch_; }

    // Getters and setters: values
    //==========================================================================

    //! Return v[i].
    //! This is NOT bound-checked.
    const value_type& operator[](size_type i) const {
      return values_[i * pitch_];
    }

    //! Return v[i].
    //! This is NOT bound-checked.
    const value_type& operator()(size_type i) const {
      return operator[](i * pitch_);
    }

    //! Get a const iterator to the beginning.
    //! NOTE: The caller must handle pitch when iterating over values.
    const_iterator begin() const {
      return values_;
    }

    //! Get a const iterator to the end.
    const_iterator end() const {
      return values_ + (n_ * pitch_);
    }

    // Operations
    //==========================================================================

    //! Multiplication with a scalar.
    dense_vector_view& operator*=(T val) {
      for (size_type i = 0; i < n_ * pitch_; i += pitch_)
        values_[i] *= val;
      return *this;
    }

    //! Division by a scalar.
    dense_vector_view& operator/=(T val) {
      assert(val != 0);
      for (size_type i = 0; i < n_ * pitch_; i += pitch_)
        values_[i] /= val;
      return *this;
    }

    // Utilities
    //==========================================================================

    // Protected data and methods
    //==========================================================================
  protected:

    using base::n_;

    //! Pointer to values.
    const_iterator values_;

    //! Pitch
    size_type pitch_;

  }; // class dense_vector_view

} // namespace sill

#endif // #ifndef _SILL_DENSE_VECTOR_VIEW_HPP_

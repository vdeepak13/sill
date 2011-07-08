
#ifndef _SILL_VECTOR_BASE_HPP_
#define _SILL_VECTOR_BASE_HPP_

#include <sill/math/sparse_linear_algebra/linear_algebra_base.hpp>
#include <sill/serialization/serialize.hpp>

namespace sill {

  /**
   * Vector base class
   *
   * @tparam T        Type of data element (e.g., float).
   * @tparam SizeType    Type of index (e.g., arma::u32).
   */
  template <typename T, typename SizeType>
  class vector_base {

    // Public types
    //==========================================================================
  public:

    typedef T                 value_type;
    typedef SizeType          size_type;
    typedef const T*          const_iterator;
    typedef T*                iterator;
    typedef const size_type*  const_index_iterator;
    typedef size_type*        index_iterator;

    // Constructors
    //==========================================================================

    //! Constructor for an empty vector.
    vector_base()
      : n_(0) { }

    //! Constructor for a vector with n elements.
    vector_base(size_type n)
      : n_(n) { }

    // Serialization
    //==========================================================================

    void save(oarchive& ar) const {
      ar << n_;
    }

    void load(iarchive& ar) {
      ar >> n_;
    }

    // Getters and setters: dimensions
    //==========================================================================

    //! Length of the vector.
    size_type size() const {
      return n_;
    }

    //! Length of the vector.
    size_type length() const {
      return size();
    }

    // Operators
    //==========================================================================

    //! Equality operator.
    bool operator==(const vector_base& other) const {
      return (n_ == other.n_);
    }

    // Protected data and methods
    //==========================================================================
  protected:

    //! Number of elements.
    size_type n_;

  }; // class vector_base

} // namespace sill

#endif // #ifndef _SILL_VECTOR_BASE_HPP_

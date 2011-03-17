
#ifndef _SILL_LINEAR_ALGEBRA_BASE_HPP_
#define _SILL_LINEAR_ALGEBRA_BASE_HPP_

namespace sill {

  /**
   * Linear algebra struct specifying types.
   *
   * @tparam T        Type of data element (e.g., float).
   * @tparam SizeType    Type of index (e.g., size_t).
   */
  template <typename T, typename SizeType>
  struct linear_algebra_base {

    // Public base
    //==========================================================================
  public:

    typedef T                 value_type;
    typedef SizeType          size_type;
    typedef const T*          const_iterator;
    typedef T*                iterator;
    typedef const size_type*  const_index_iterator;
    typedef size_type*        index_iterator;

  }; // struct linear_algebra_base

} // namespace sill

#endif // #ifndef _SILL_LINEAR_ALGEBRA_BASE_HPP_

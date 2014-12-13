#ifndef SILL_COPY_PTR_HPP
#define SILL_COPY_PTR_HPP

#include <boost/serialization/shared_ptr.hpp>
#include <boost/pointee.hpp>

#include <sill/stl_concepts.hpp>
#include <sill/serialization/serialize.hpp>
#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A shared pointer that performs garbage collection via reference
   * counting and implements copy-on-write semantics.  The type
   * supplied must be copy constructible.  If the pointer is dereferenced
   * with a const access, a const-reference to the underlying object
   * is returned (in other words, unlike standard pointers, copy_ptr
   * propagates constness). If the pointer is dereferenced with
   * a mutable access, and if the object is shared by more than
   * one pointer, it is first copied. This semantics ensure that several
   * pointers each have independent views of the shared object, while
   * avoiding unnecessary copies of the object.
   *
   * The current implementation is not thread-safe.
   *
   * @tparam T the pointed type. T must be CopyConstructible
   */
  template <typename T>
  class copy_ptr {
    concept_assert((CopyConstructible<T>));

    //! The underlying pointer, which performs reference counting.
    mutable typename boost::shared_ptr<T> rc_ptr;

    friend class boost::serialization::access;


    /**
     * If this pointer does not have unique ownership of the object,
     * then a copy of the object is created and this pointer is
     * updated.  This function can be used in preparation for an
     * update to the object.
     *
     * One might think that this method is too conservative, in that
     * if only const_ptr_t pointers share this object with this copy_ptr
     * pointer, then it is alright to modify the object without making
     * a copy.  But this would alter the view of the const pointers,
     * and to them it would appear the object changed without cause.
     * So it is correct to use copy-on-write whenever the shared
     * pointer is not unique.
     */
    void copy() const {
      if (!rc_ptr.unique())
	    rc_ptr.reset(new T(*rc_ptr));
    }

  public:

    //! Construction of the null pointer.
    copy_ptr() : rc_ptr() { }

    //! Explicit construction from raw pointer returned by new().
    explicit copy_ptr(T* raw_ptr) : rc_ptr(raw_ptr) { }

    //! Construction from a reference-counted pointer.
    explicit copy_ptr(const typename boost::shared_ptr<T>& rc_ptr)
      : rc_ptr(rc_ptr) { }

    //! Copy constructor.
    copy_ptr(const copy_ptr& ptr) : rc_ptr(ptr.rc_ptr) {
//      if (this->rc_ptr)
//        copy();
    }

    //! Validity test.
    operator bool() const { return rc_ptr; }

    //! Reset to the null pointer.
    void reset() { rc_ptr.reset(); }

    //! Resets to own a new raw pointer.
    template <class CompatibleT> 
    void reset(CompatibleT* raw_ptr) {
      rc_ptr.reset(raw_ptr); 
    }

    //! Assignment.
    const copy_ptr& operator=(const copy_ptr& ptr) {
      this->rc_ptr = ptr.rc_ptr;
//      if (this->rc_ptr)
//        copy();
      return *this;
    }

    //! Swaps this pointer with the supplied pointer.
    void swap(copy_ptr& other) { rc_ptr.swap(other.rc_ptr); }

    //! Dereference.
    T& operator*() { copy(); return *rc_ptr; }
    const T& operator*() const { return *rc_ptr; }

    //! Pointer access.
    T* operator->() { copy(); return rc_ptr.operator->(); }
    const T* operator->() const { return rc_ptr.operator->(); }

    //! Equality operator.
    bool operator==(const copy_ptr& ptr) const {
      return (this->rc_ptr == ptr.rc_ptr);
    }
    //! Inequality operator.
    bool operator!=(const copy_ptr& ptr) const {
      return (this->rc_ptr != ptr.rc_ptr);
    }

    //! Less than comparison of the underlying shared pointer
    bool operator<(const copy_ptr& ptr) const {
      return (this->rc_ptr < ptr.rc_ptr);
    }

    //! Returns (use_count()==1).
    bool unique() const { return rc_ptr.unique(); }

    //! Returns the number of objects, *this included, that share ownership
    //! with *this, or an unspecified nonnegative value when *this* is empty.
    long use_count() const { return rc_ptr.use_count(); }

    //! Serialize / deserialize members
    void serialize(oarchive& ar) const {
      ar << *rc_ptr;
    }
    void deserialize(iarchive& ar) {
      ar >> *rc_ptr;
    }

  }; // class copy_ptr<T>

} // namespace sill

namespace boost {                            
namespace serialization {                    
template<typename T>                                   
struct tracking_level< sill::copy_ptr<T> >                   
{                                            
    typedef mpl::integral_c_tag tag;         
    typedef mpl::int_<track_never> type;              
    BOOST_STATIC_CONSTANT(                   
        int,                                 
        value = tracking_level::type::value  
    );                                       
    /* tracking for a class  */              
    BOOST_STATIC_ASSERT((                    
        mpl::greater<                        
            /* that is a prmitive */         
        implementation_level< sill::copy_ptr<T> >,       
            mpl::int_<primitive_type>        
        >::value                             
    ));                                      
};                                           
}}

namespace boost {

  /**
   * A traits declaration so the Boost library can infer the types
   * associated with SILL pointers.
   */
  template <typename T>
  struct pointee<sill::copy_ptr<T> > { typedef T type; };

} // namespace boost

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_POINTER_HPP

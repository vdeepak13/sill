
#ifndef PRL_ALLOCATOR_HPP
#define PRL_ALLOCATOR_HPP

#include <iterator>
#include <limits>

namespace prl {

  /**
   * An allocator adaptor that statically allocates some number of
   * elements in advance to avoid the overhead of dynamic memory
   * allocation.
   *
   * Note: When standard containers are swapped, their allocators are _not_
   *       swapped. Therefore, you should attempt to use the the swap function
   *       on objects that use this allocator.
   */
  template <typename T,
	    std::size_t pre_alloc_size = 0,
	    typename Allocator = std::allocator<T> >
  class pre_allocator {

  protected:

    //! The pre-allocated elements.
    T elts[pre_alloc_size];

    //! A flag indicating if the pre-allocation has been used.
    bool pre_alloc_used;

  public:
 
    typedef T value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    template<typename new_elt_t>
    struct rebind {
      typedef pre_allocator<new_elt_t, pre_alloc_size, Allocator> other;
    };

    // Constructors.
    explicit pre_allocator() { pre_alloc_used = false; }
    ~pre_allocator() {}
     pre_allocator(const pre_allocator&) { 
      pre_alloc_used = false; 
    }
    template <typename U>
    explicit pre_allocator(const prl::pre_allocator<U, pre_alloc_size, Allocator>&) {
      pre_alloc_used = false;
    }

    // Addresses.  (Notice these will fail if operator& is overloaded.)
    pointer address(reference r) { return &r; }
    const_pointer address(const_reference r) { return &r; }

    // Memory allocation.
    pointer allocate(size_type cnt, 
			    typename std::allocator<void>::const_pointer = 0) {
      if (!pre_alloc_used && (cnt <= pre_alloc_size)) {
	pre_alloc_used = true;
	return elts;
      } else
	return reinterpret_cast<pointer>(::operator new(cnt * sizeof (T))); 
    }
    void deallocate(pointer p, size_type) { 
      if (p == elts)
	pre_alloc_used = false;
      else
        ::operator delete(p); 
    }

    //    size
    size_type max_size() const { 
        return std::numeric_limits<size_type>::max() / sizeof(T);
 }

    //    construction/destruction
    void construct(pointer p, const T& t) { new(p) T(t); }
    void destroy(pointer p) { p->~T(); }

    bool operator==(pre_allocator const&) { return true; }
    bool operator!=(pre_allocator const& a) { return !operator==(a); }

  }; // pre_allocator

} // end of namespace: prl

#endif // #ifndef PRL_ALLOCATOR_HPP

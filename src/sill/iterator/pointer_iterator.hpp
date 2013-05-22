#ifndef SILL_POINTER_ITERATOR_HPP
#define SILL_POINTER_ITERATOR_HPP

#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/iterator_categories.hpp>

namespace sill {

  /** 
   * A class that wraps around the raw pointer T* and provides member 
   * functions of iterator_category. It could also be easily extended
   * to do boundary checking.
   *
   * \ingroup iterator
   */
  template <typename T, typename Category>
  class pointer_iterator 
    : public boost::iterator_facade<pointer_iterator<T, Category>, T, Category>
  {
  private: 
    typedef 
      boost::iterator_facade<pointer_iterator<T, Category>, T, Category> base;
      
    T* ptr;
    
  public:
    typedef typename base::value_type value_type;
    typedef typename base::difference_type difference_type;

    pointer_iterator(T* ptr) : ptr(ptr) { }

    template <typename X>
    pointer_iterator(X* ptr) : ptr(ptr) { }

  private:
    friend class boost::iterator_core_access;

    T& dereference() const { return *ptr; }
    
    template <typename X>
    bool equal(pointer_iterator<X, Category> it) const {
      return ptr == it.ptr; 
    }

    void increment() { ptr++; }
    void decrement() { ptr--; }
    void advance(difference_type n) { ptr += n; }

    template <typename X>
    difference_type distance_to(pointer_iterator<X, Category> it) const { 
      return (it.ptr) - ptr; 
    }
  };

  // todo: simplify. provide index()
  
} // namespace sill

#endif 

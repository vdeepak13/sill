#ifndef PRL_FORWARD_ITERATOR_HPP
#define PRL_FORWARD_ITERATOR_HPP

#include <iterator>
#include <typeinfo>

#include <boost/type_traits/remove_reference.hpp>
#include <boost/type_traits/remove_const.hpp>

#include <prl/global.hpp>

namespace prl {
  
  /**
   * A polymorphic forward iterator.
   * \ingroup iterator
   */
  template <typename Ref>
  class forward_iterator {

    // Public type declarations
    //==========================================================================
  public:
    typedef std::forward_iterator_tag                     iterator_category;
    typedef std::ptrdiff_t                                difference_type;
    typedef Ref                                           reference;
    typedef typename boost::remove_reference<Ref>*        pointer;
    typedef typename boost::remove_const<
      typename boost::remove_reference<Ref>::type>::type  value_type;


    // Private type declarations and data members
    //==========================================================================
  private:
    //! The base class for iterators that hold 
    struct abstract_placeholder {
      virtual Ref dereference() const = 0;
      virtual void increment() = 0;
      virtual bool equal(const abstract_placeholder& other) const = 0;
      virtual abstract_placeholder* clone() const = 0;
      virtual ~abstract_placeholder() { }
    };
    
    //! A placeholder for a specific iterator
    template <typename It>
    class placeholder : public abstract_placeholder {
      It it;
    public:
      placeholder(const It& it) : it(it) { }
      Ref dereference() const { return *it; }
      void increment() { ++it; }
      bool equal(const abstract_placeholder& other) const {
        assert(typeid(*this) == typeid(other));
        return it == static_cast<const placeholder&>(other).it;
      }
      abstract_placeholder* clone() const {
        return new placeholder(*this);
      }
    };
    
    abstract_placeholder* p;

    // Constructors and destructors
    //==========================================================================
  public:
    //! Default constructor; leaves the iterator uninitialized
    forward_iterator() : p() { }

    //! Copy constructor
    forward_iterator(const forward_iterator& other) 
      : p(other.p->clone()) { }

    //! Encapsulates an arbitrary iterator
    template <typename It>
    explicit forward_iterator(const It& it) 
      : p(new placeholder<It>(it)) { }

    //! Destructor
    ~forward_iterator() {
      if (p) delete p;
    }

    //! Assignment operator
    forward_iterator& operator=(const forward_iterator& other) {
      if (p) delete p;
      p = other.p->clone();
      return *this;
    }

    //! Swaps two forward_iterators
    void swap(forward_iterator& other) {
      std::swap(p, other.p);
    }
    
    // Iterator operations
    //==========================================================================
    Ref operator*() const {
      return p->dereference();
    }
    
    forward_iterator& operator++() {
      p->increment();
      return *this;
    }

    forward_iterator operator++(int) {
      forward_iterator tmp(*this);
      ++*this;
      return tmp;
    }

    bool operator==(const forward_iterator& other) const {
      return p->equal(*other.p);
    }

    bool operator!=(const forward_iterator& other) const {
      return !p->equal(*other.p);
    }
    
  }; // class forward_iterator

} // namespace prl


#endif

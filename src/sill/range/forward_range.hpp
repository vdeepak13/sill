#ifndef SILL_FORWARD_RANGE_HPP
#define SILL_FORWARD_RANGE_HPP

#include <iterator>
#include <utility> // std::pair
#include <vector>
#include <list>
#include <set> 

#include <boost/array.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/range/iterator_range.hpp>

#include <sill/iterator/forward_iterator.hpp>
#include <sill/math/linear_algebra/armadillo.hpp>

// Forward declaration
namespace arma {
  template <typename T> class Col;
}

namespace sill {

  using boost::enable_if;

  /**
   * A range that implements type erasure to represent a reference 
   * to any forward range.
   * \ingroup range_adapters
   */
  template <typename Ref>
  class forward_range 
    : public boost::iterator_range< forward_iterator<Ref> > {

    typedef boost::iterator_range< forward_iterator<Ref> > base;

    //! A class that specifies whether an iterator is compatible with this range
    template <typename It>
    struct is_compatible_iterator
      : boost::is_convertible<typename std::iterator_traits<It>::reference, Ref>
    { };
      
    //! A class that specifies whether a reference is compatible with this range
    template <typename Ref2>
    struct is_compatible_ref
      : boost::is_convertible<Ref2, Ref> { };

    // Constructors
    //==========================================================================
  public:
    
    //! Constructs a forward_range from a boost::iterator_range
    template <typename It>
    forward_range(const boost::iterator_range<It>& range,
                  typename enable_if< is_compatible_iterator<It> >::type* = 0) 
      : base(range) { }
    
    //! Constructs a forward_range from a pair of iterators
    template <typename It>
    forward_range(const std::pair<It,It>& range,
                  typename enable_if< is_compatible_iterator<It> >::type* = 0)
      : base(range) { } 

    //! Construct a forward_range from a source iterator and a dest iterator
    template <typename It>
    forward_range(const It& begin, const It& end,
                  typename enable_if< is_compatible_iterator<It> >::type* = 0)
      : base( std::pair<It,It>(begin, end) ) { } 

    //! Constructs a forward_range from std::vector
    template <typename T>
    forward_range(const std::vector<T>& range,
                  typename enable_if< is_compatible_ref<const T&> >::type* = 0)
      : base(range) { }
    
    //! Constructs a forward_range from std::list
    template <typename T>
    forward_range(const std::list<T>& range,
                  typename enable_if< is_compatible_ref<const T&> >::type* = 0)
      : base(range) { }
    
    //! Constructs a forward_range from std::set
    template <typename T>
    forward_range(const std::set<T>& range,
                  typename enable_if< is_compatible_ref<const T&> >::type* = 0)
      : base(range) { }

    //! Constructs a forward_range from boost::array
    template <typename T, std::size_t N>
    forward_range(const boost::array<T, N>& range,
                  typename enable_if< is_compatible_ref<const T&> >::type* = 0)
      : base(range) { }

    //! Constructs a forward_range from arma::Col
    template <typename T>
    forward_range(const arma::Col<T>& range,
                  typename enable_if< is_compatible_ref<const T&> >::type* = 0)
      : base(range) { }

  }; // class forward_range
  
} // namespace sill

#endif

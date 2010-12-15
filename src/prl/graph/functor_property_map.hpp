#ifndef PRL_PROPERTY_MAP
#define PRL_PROPERTY_MAP

#include <boost/property_map/property_map.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/type_traits/remove_reference.hpp>

#include <prl/stl_concepts.hpp>

#include <prl/global.hpp>

namespace prl {

  //! A readable property map based on an unary function
  //! @see Boost.PropertyMap
  template <typename F>
  class functor_property_map {

    F f;

  public:
    typedef boost::readable_property_map_tag  category;
    typedef typename F::argument_type         key_type;
    typedef typename F::result_type           reference;
    typedef typename boost::remove_const<
      typename boost::remove_reference<reference>::type>::type value_type;

    functor_property_map(F f) : f(f){ }

    value_type operator[](key_type key) const {
      return f(key);
    }

  }; // class functor_property_map

  //! Implements the PropertyMap concept
  //! \relates functor_property_map
  template<typename F>
  typename F::result_type get(const functor_property_map<F>& map,
                              typename F::argument_type key) {
    return map[key];
  }
  
  //! A readable property map based on a functor
  //! \relates functor_property_map
  template <typename F>
  functor_property_map<F> make_functor_property_map(const F& f) {
    return functor_property_map<F>(f);
  }

} // namespace prl

#endif 

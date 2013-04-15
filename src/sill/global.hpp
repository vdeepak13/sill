
#ifndef SILL_GLOBAL_HPP
#define SILL_GLOBAL_HPP

#include <cassert>
#include <iosfwd> 
#include <utility>

#include <boost/mpl/void.hpp> // for boost::mpl::void_
#include <boost/tuple/tuple.hpp> // for boost::tie

namespace sill {
  class iarchive;
  class oarchive;
}

BOOST_MPL_AUX_ADL_BARRIER_NAMESPACE_OPEN

  // For some reason, if we place this operator in sill::, it won't be found
  inline std::ostream& operator<<(std::ostream& out, void_) {
    return out;
  }

  inline sill::oarchive& operator<<(sill::oarchive& ar, const void_&) {
    return ar;
  }

  inline sill::iarchive& operator>>(sill::iarchive& ar, void_&) {
    return ar;
  }

BOOST_MPL_AUX_ADL_BARRIER_NAMESPACE_CLOSE


namespace sill {

  // standard type to represent size
  using std::size_t;

  // void declaration
  using boost::mpl::void_;

  //! Type deduction will fail unless the arguments have the same type.
  template <typename T> void same_type(const T&, const T&) { }

  //! Equality comparison for void_
  inline bool operator==(const void_&, const void_&) {
    return true;
  }

  //! Inequality comparison for void_
  inline bool operator!=(const void_&, const void_&) {
    return false;
  }

} // namespace sill

#endif 

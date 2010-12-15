
#ifndef PRL_GLOBAL_HPP
#define PRL_GLOBAL_HPP

#include <cassert>
#include <iosfwd> 
#include <utility>

#include <boost/mpl/void.hpp> // for boost::mpl::void_
#include <boost/tuple/tuple.hpp> // for boost::tie

namespace prl {
  class iarchive;
  class oarchive;
}

BOOST_MPL_AUX_ADL_BARRIER_NAMESPACE_OPEN

  // For some reason, if we place this operator in prl::, it won't be found
  inline std::ostream& operator<<(std::ostream& out, void_) {
    return out;
  }

  inline prl::oarchive& operator<<(prl::oarchive& ar, const void_&) {
    return ar;
  }

  inline prl::iarchive& operator>>(prl::iarchive& ar, void_&) {
    return ar;
  }

BOOST_MPL_AUX_ADL_BARRIER_NAMESPACE_CLOSE


namespace prl {

  // standard type to represent size
  using std::size_t;

  // void declaration
  using boost::mpl::void_;

  //! Type deduction will fail unless the arguments have the same type.
  template <typename T> void same_type(const T&, const T&) { }

} // namespace prl

#endif 

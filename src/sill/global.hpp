#ifndef SILL_GLOBAL_HPP
#define SILL_GLOBAL_HPP

#include <cassert>
#include <cmath>
#include <iosfwd> 
#include <utility>

#include <boost/tuple/tuple.hpp> // for boost::tie

namespace sill {
  class iarchive;
  class oarchive;
}

namespace sill {

  // standard type to represent size
  using std::size_t;

  // standard type to represent difference
  using std::ptrdiff_t;

  // bring in log and exp to allow uniform handling of T and logarithmic<T>
  using std::log;
  using std::exp;

  //! Type deduction will fail unless the arguments have the same type.
  template <typename T> void same_type(const T&, const T&) { }

  // empty type (useful primarily for empty vertex and edge properties)
  struct void_ { };

  //! equality comparison for void_
  inline bool operator==(const void_&, const void_&) {
    return true;
  }

  //! inequality comparison for void_
  inline bool operator!=(const void_&, const void_&) {
    return false;
  }
  
  //! less than comparison for void_
  inline bool operator<(const void_&, const void_&) {
    return false;
  }

  //! i/o
  inline std::ostream& operator<<(std::ostream& out, void_) {
    return out;
  }

  //! serialization
  inline sill::oarchive& operator<<(sill::oarchive& ar, const void_&) {
    return ar;
  }

  //! deserialization
  inline sill::iarchive& operator>>(sill::iarchive& ar, void_&) {
    return ar;
  }

} // namespace sill

#endif 

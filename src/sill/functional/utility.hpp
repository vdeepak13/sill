#ifndef SILL_FUNCTIONAL_UTILITY_HPP
#define SILL_FUNCTIONAL_UTILITY_HPP

#include <utility>

namespace sill {

  //! Returns the first value of a pair.
  struct pair_first {
    template <typename T, typename U>
    T operator()(const std::pair<T, U>& value) const {
      return value.first;
    }
  };

  //! Returns the second value of a pair.
  struct pair_second {
    template <typename T, typename U>
    U operator()(const std::pair<T, U>& value) const {
      return value.second;
    }
  };

  //! Returns true if the first contained is smaller than the second one.
  struct size_less {
    template <typename Container>
    bool operator()(const Container& x, const Container& y) const {
      return x.size() < y.size();
    }
  };

  //! Returns true if the first contained is larger than the second one.
  struct size_greater {
    template <typename Container>
    bool operator()(const Container& x, const Container& y) const {
      return x.size() > y.size();
    }
  };

} // namespace sill

#endif
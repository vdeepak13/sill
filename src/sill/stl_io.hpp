#ifndef SILL_STL_IO_HPP
#define SILL_STL_IO_HPP

#include <iostream>
#include <utility> // for std::pair
#include <vector>
#include <list>
#include <set>
#include <map>
#include <algorithm>

#include <sill/range/concepts.hpp>

#include <sill/macros_def.hpp>

// forward declaration
namespace boost {

  template <typename T, std::size_t N> class array;
}

namespace sill 
{

  // Implementation of output functions

  template <typename Char, typename Traits, typename First, typename Second>
  std::basic_ostream<Char,Traits>&
  operator<<(std::basic_ostream<Char,Traits>& out,
	     const std::pair<First,Second>& p) {
    out << "<" << p.first << "," << p.second << ">";
    return out;
  }

  template <typename Char, typename Traits, typename Range>
  std::basic_ostream<Char, Traits>&
  print_range(std::basic_ostream<Char,Traits>& out, 
	      const Range& values,
	      char opening, char delimiter, char closing) {
    concept_assert((InputRange<Range>));
    typename Range::const_iterator it, end;
    out << opening;
    for(it = values.begin(), end = values.end(); it != end;) {
      out << *it;
      if (++it != end) out << delimiter;
    }
    out << closing;
    return out;
  }

  #define SILL_PRINT_CONTAINER1(container, opening, delimiter, closing)	\
  template <typename Char, typename Traits,				\
	    typename A1, typename Allocator>				\
  std::basic_ostream<Char,Traits>&					\
  operator<<(std::basic_ostream<Char,Traits>& out,			\
	     const container<A1, Allocator>& c) {			\
    return print_range(out, c, opening, delimiter, closing);		\
  }
  
  #define SILL_PRINT_CONTAINER2(container, opening, delimiter, closing)	\
  template <typename Char, typename Traits,				\
	    typename A1, typename A2, typename Allocator>		\
  std::basic_ostream<Char,Traits>&					\
  operator<<(std::basic_ostream<Char,Traits>& out,			\
	     const container<A1, A2, Allocator>& c) {			\
    return print_range(out, c, opening, delimiter, closing);		\
  }
  
  #define SILL_PRINT_CONTAINER3(container, opening, delimiter, closing)	\
  template <typename Char, typename Traits,				\
	    typename A1, typename A2, typename A3, typename Allocator>	\
  std::basic_ostream<Char,Traits>&					\
  operator<<(std::basic_ostream<Char,Traits>& out,			\
	     const container<A1, A2, A3, Allocator>& c) {		\
    return print_range(out, c, opening, delimiter, closing);		\
  }
  
  SILL_PRINT_CONTAINER2(std::set, '{', ' ', '}');
  SILL_PRINT_CONTAINER2(std::multiset, '{', ' ', '}');
  SILL_PRINT_CONTAINER3(std::map, '{', ' ', '}');
  SILL_PRINT_CONTAINER3(std::multimap, '{', ' ', '}');
  SILL_PRINT_CONTAINER1(std::vector, '[', ' ', ']');
  SILL_PRINT_CONTAINER1(std::list, '(', ' ', ')');
  
  #undef SILL_PRINT_CONTAINER1
  #undef SILL_PRINT_CONTAINER2
  #undef SILL_PRINT_CONTAINER3

  template <typename Char, typename Traits, typename T, std::size_t N>
  std::basic_ostream<Char,Traits>&
  operator<<(std::basic_ostream<Char,Traits>& out,
	     const boost::array<T,N>& a) {
    return print_range(out, a, '[', ' ', ']');
  }

  // Implementation of input functions

  //! Read in a pair <val1,val2>
  //! \todo Can we overload operator<< for this?  I tried but didn't get it to
  //!       work.
  template <typename Char, typename Traits, typename First, typename Second>
  static void read_pair(std::basic_istream<Char,Traits>& in,
                        std::pair<First,Second>& p) {
    char c;
    in.get(c);
    if (c == ' ')
      in.get(c);
    assert(c == '<');
    if (!(in >> p.first))
      assert(false);
    in.get(c);
    assert(c == ',');
    if (!(in >> p.second))
      assert(false);
    in.get(c);
    assert(c == '>');
  }

  //! Read in a vector of values [val1,val2,...], ignoring an initial space
  //! if necessary.
  //! \todo Can we overload operator<< for this?  I tried but didn't get it to
  //!       work.
  template <typename T>
  static void read_vec(std::istream& in, std::vector<T>& v) {
    char c;
    T val;
    v.clear();
    in.get(c);
    if (c == ' ')
      in.get(c);
    assert(c == '[');
    if (in.peek() != ']') {
      do {
        if (!(in >> val))
          assert(false);
        v.push_back(val);
        if (in.peek() == ',')
          in.ignore(1);
      } while (in.peek() != ']');
    }
    in.ignore(1);
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

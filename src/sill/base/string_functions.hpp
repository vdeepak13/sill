
#ifndef SILL_STRING_FUNCTIONS_HPP
#define SILL_STRING_FUNCTIONS_HPP

#include <cassert>
#include <sstream>
#include <string>
#include <vector>

#include <sill/range/forward_range.hpp>

namespace sill {

  /**
   * Splits a file path into a directory and the file name.
   * E.g., "directory/file.txt" --> "directory" "file.txt"
   *       "file.txt"           --> "."         "file.txt"
   * @return <directory, filename>
   */
  std::pair<std::string, std::string>
  split_directory_file(const std::string& filepath);

  //! @return argument printed to a string
  template<typename T>
  std::string to_string(const T& t) {
    std::ostringstream o;
    o << t;
    return o.str();
  }

  //! Convert all letters in a string to lowercase.
  std::string tolower(const std::string& s);

  //! Convert all letters in a string to uppercase.
  std::string toupper(const std::string& s);

  //! Convert all letters in a string to lowercase in place.
  void tolower_inplace(std::string& s);

  //! Convert all letters in a string to uppercase in place.
  void toupper_inplace(std::string& s);

  //! Remove leading and trailing whitespace.
  std::string chop_whitespace(const std::string& s);

  //! Convert all instances of character a in string s to character b in place.
  void swap_characters_inplace(std::string& s, char a, char b);

  //! Convert all non-alphanumeric characters in string s to character c,
  //! in place.
  void nonalnum_to_c_inplace(std::string& s, char c);

  /**
   * Concatenate the given vector of values (using operator<< to print them),
   * with sep separating them.  (Just like Perl join.)
   * @param  sep   String with which to separate the values.
   * @param  vals  Values to concatenate.
   */
  template <typename VecType>
  std::string string_join(const std::string& sep, const VecType& vals) {
    if (vals.size() == 0)
      return std::string();
    std::ostringstream o;
    for (size_t i = 0; i < vals.size() - 1; ++i)
      o << vals[i] << sep;
    o << vals[vals.size() - 1];
    return o.str();
  }

  /**
   * Concatenate the given range of values (using operator<< to print them),
   * with sep separating them.  (Just like Perl join.)
   * @param  sep   String with which to separate the values.
   * @param  vals  Values to concatenate.
   */
  template <typename T>
  std::string
  string_join(const std::string& sep, const forward_range<T>& vals) {
    typename forward_range<T>::const_iterator it = vals.begin();
    typename forward_range<T>::const_iterator end = vals.end();
    if (it == end)
      return std::string();
    std::ostringstream o;
    while (it != end) {
      o << *it;
      ++it;
      if (it != end)
        o << sep;
    }
    return o.str();
  }

  /**
   * Parse the given string as a value of type T.
   * Assert false if parsing fails.
   */
  template <typename T>
  T parse_string(const std::string& s) {
    std::istringstream is(s);
    T val = T();
    if (!(is >> val))
      assert(false);
    return val;
  }

} // namespace sill

#endif // #ifndef SILL_STRING_FUNCTIONS_HPP

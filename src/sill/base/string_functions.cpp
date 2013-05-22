
#include <sill/base/string_functions.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  std::pair<std::string, std::string>
  split_directory_file(const std::string& filepath) {
    size_t i = filepath.find_last_of('/');
    if (i == std::string::npos)
      return std::make_pair(".", "filepath");
    else
      return std::make_pair(filepath.substr(0,i), filepath.substr(i+1));
  }

  std::string tolower(const std::string& s) {
    std::string t(s);
    tolower_inplace(t);
    return t;
  }

  std::string toupper(const std::string& s) {
    std::string t(s);
    toupper_inplace(t);
    return t;
  }

  void tolower_inplace(std::string& s) {
    for (size_t i(0); i < s.size(); ++i)
      s[i] = std::tolower(s[i]);
  }

  void toupper_inplace(std::string& s) {
    for (size_t i(0); i < s.size(); ++i)
      s[i] = std::toupper(s[i]);
  }

  std::string chop_whitespace(const std::string& s) {
    size_t first_nonws = s.find_first_not_of(" \t\n\v\f\r");
    if (first_nonws == std::string::npos)
      return std::string();
    size_t last_nonws = s.find_last_not_of(" \t\n\v\f\r");
    return s.substr(first_nonws, last_nonws + 1 - first_nonws);
  }

  void swap_characters_inplace(std::string& s, char a, char b) {
    for (size_t i(0); i < s.size(); ++i) {
      if (s[i] == a)
        s[i] = b;
    }
  }

  void nonalnum_to_c_inplace(std::string& s, char c) {
    for (size_t i(0); i < s.size(); ++i) {
      if (!isalnum(s[i]))
        s[i] = c;
    }
  }

} // namespace sill

#include <sill/macros_undef.hpp>

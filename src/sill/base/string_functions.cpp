
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

  void tolower_inplace(std::string& s) {
    for (size_t i(0); i < s.size(); ++i)
      s[i] = std::tolower(s[i]);
  }

  void toupper_inplace(std::string& s) {
    for (size_t i(0); i < s.size(); ++i)
      s[i] = std::toupper(s[i]);
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

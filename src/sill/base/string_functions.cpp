
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

} // namespace sill

#include <sill/macros_undef.hpp>

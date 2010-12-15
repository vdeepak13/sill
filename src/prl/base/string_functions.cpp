
#include <prl/base/string_functions.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  std::pair<std::string, std::string>
  split_directory_file(const std::string& filepath) {
    size_t i = filepath.find_last_of('/');
    if (i == std::string::npos)
      return std::make_pair(".", "filepath");
    else
      return std::make_pair(filepath.substr(0,i), filepath.substr(i+1));
  }

} // namespace prl

#include <prl/macros_undef.hpp>

#ifndef SILL_LINE_SEARCH_FAILED_HPP
#define SILL_LINE_SEARCH_FAILED_HPP

#include <stdexcept>

namespace sill {

  class line_search_failed : public std::runtime_error {
  public:
    line_search_failed(const std::string& reason)
      : std::runtime_error(reason) { }
  };

} // namespace sill

#endif

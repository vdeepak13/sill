#ifndef SILL_NUMERICAL_ERROR
#define SILL_NUMERICAL_ERROR

#include <stdexcept>

namespace sill {

  struct numerical_error : public std::runtime_error {
    explicit numerical_error(const std::string& msg)
      : runtime_error(msg) { }
    explicit numerical_error(const char* msg)
      : runtime_error(msg) { }
  };

} // namespace sill

#endif

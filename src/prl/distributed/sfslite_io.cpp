#include <iostream>

#include <prl/distributed/sfslite_io.hpp>

#include <str.h>
#include <bigint.h>

std::ostream& operator<<(std::ostream& out, const str& s) {
  out << s.cstr();
  return out;
}

std::ostream& operator<<(std::ostream& out, const bigint& b) {
  out << b.getstr().cstr();
  // bigint's cstr() function does not seem safe
  return out;
}

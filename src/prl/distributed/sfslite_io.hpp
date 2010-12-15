#ifndef PRL_SFSLITE_IO_HPP
#define PRL_SFSLITE_IO_HPP

#include <iosfwd>

// forward declarations
class str;
class bigint;

//! Print a string
std::ostream& operator<<(std::ostream& out, const str& s);

//! Print a bigint in hexadecimal notation
std::ostream& operator<<(std::ostream& out, const bigint& b);

#endif

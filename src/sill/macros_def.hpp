// Defines some useful macros that emulate C++0x language extensions
//   concept_assert, foreach
// See model/
// Note: this file does not have header-guards to allow multiple (def-undef)s

#include <boost/foreach.hpp>
#include <boost/static_assert.hpp>

// Shortcut macro definitions
//! see http://www.boost.org/doc/html/foreach.html
#define foreach BOOST_FOREACH
#define revforeach BOOST_REVERSE_FOREACH

//! use Boost concept checking framework to check concepts
#ifdef SILL_CHECK_CONCEPTS
  #include <boost/concept_check.hpp>
  #define concept_assert BOOST_CONCEPT_ASSERT
  #define concept_usage BOOST_CONCEPT_USAGE
#else
  #define concept_assert(X)
  #define concept_usage(X) void usage()
#endif

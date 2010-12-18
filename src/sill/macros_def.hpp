// Defines some useful macros that emulate C++0x language extensions
//   static_assert, concept_assert, foreach
// See model/
// Note: this file does not have header-guards to allow multiple (def-undef)s

#include <boost/foreach.hpp>
#include <boost/static_assert.hpp>
#include <boost/concept_check.hpp>
#include <boost/serialization/nvp.hpp>

// Shortcut macro definitions
//! see http://www.boost.org/doc/html/foreach.html
#define foreach BOOST_FOREACH
#define revforeach BOOST_REVERSE_FOREACH

//! see http://www.boost.org/doc/html/boost_staticassert.html
#define static_assert BOOST_STATIC_ASSERT

//! new Boost concept checking framework
#define concept_assert BOOST_CONCEPT_ASSERT

//! new Boost concept checking framework
#define concept_usage BOOST_CONCEPT_USAGE

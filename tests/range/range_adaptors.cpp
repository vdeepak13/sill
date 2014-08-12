#define BOOST_TEST_MODULE range_adaptors
#include <boost/test/unit_test.hpp>

#include <sill/range/transformed.hpp>
#include <sill/range/reversed.hpp>
#include <sill/range/joined.hpp>
#include <sill/range/forward_range.hpp>

#include <boost/array.hpp>

#include <sill/macros_def.hpp>

template <size_t N, typename Range>
bool compare(const boost::array<int, N>& array, const Range& adapted) {
  size_t i = 0;
  foreach(const int& x, adapted) {
    if (array[i++] != x) {
      return false;
    }
  }
  return i == N;
}

int plus_one(int x) {
  return x + 1;
}

BOOST_AUTO_TEST_CASE(test_all) {
  using namespace sill;

  boost::array<int,4> a = {{0, 2, 4, 8}};
  boost::array<int,2> b = {{0, 2}};

  boost::array<int,4> a1 = {{1, 3, 5, 9}};
  boost::array<int,4> ar = {{8, 4, 2, 0}};
  boost::array<int,6> ab = {{0, 2, 4, 8, 0, 2}};

  BOOST_CHECK(compare(a1, make_transformed(a, plus_one)));
  BOOST_CHECK(compare(ar, make_reversed(a)));
  BOOST_CHECK(compare(ab, make_joined(a, b)));
  BOOST_CHECK(compare(a, forward_range<int>(a)));
}

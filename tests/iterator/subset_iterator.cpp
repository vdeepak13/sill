#define BOOST_TEST_MODULE subset_iterator
#include <boost/test/unit_test.hpp>

#include <sill/iterator/subset_iterator.hpp>

#include <set>

typedef std::multiset<std::set<char> > char_set_multiset;
typedef sill::subset_iterator<std::set<char> > char_set_iterator;

BOOST_TEST_DONT_PRINT_LOG_VALUE(char_set_multiset);

char_set_multiset hardcoded(size_t min, size_t max) {
  char_set_multiset result;
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      for (int c = 0; c < 2; ++c) {
        for (int d = 0; d < 2; ++d) {
          size_t sum = a + b + c + d;
          if (min <= sum && sum <= max) {
            std::set<char> elem;
            if (a) elem.insert('a');
            if (b) elem.insert('b');
            if (c) elem.insert('c');
            if (d) elem.insert('d');
            result.insert(elem);
          }
        }
      }
    }
  }
  return result;
}

char_set_multiset iterated(char_set_iterator it) {
  char_set_multiset result;
  char_set_iterator end;
  for (; it != end; ++it) {
    result.insert(*it);
  }
  return result;
}

BOOST_AUTO_TEST_CASE(test_all) {
  std::set<char> s;
  s.insert('a');
  s.insert('b');
  s.insert('c');
  s.insert('d');

  // strict subsets, minus the empty set
  BOOST_CHECK_EQUAL(hardcoded(1, 3), iterated(char_set_iterator(s, 1, 3)));

  // all subsets
  BOOST_CHECK_EQUAL(hardcoded(0, 4), iterated(char_set_iterator(s)));

  // subsets of size <= 2
  BOOST_CHECK_EQUAL(hardcoded(0, 2), iterated(char_set_iterator(s, 0, 2)));
}

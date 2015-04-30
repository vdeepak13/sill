#define BOOST_TEST_MODULE set_index
#include <boost/test/unit_test.hpp>

#include <sill/datastructure/set_index.hpp>
#include <sill/iterator/counting_output_iterator.hpp>

#include <boost/range/algorithm.hpp>

#include <iostream>
#include <random>
#include <set>
#include <vector>

namespace sill {
  template class set_index<int, std::vector<double>>;
}

using namespace sill;
size_t n = 200;

struct fixture {
  typedef std::list<int> list_type;
  typedef std::set<int> set_type;

  fixture() {
    std::mt19937 rng;
    for (size_t i = 0; i < n; ++i) {
      size_t size = 1 + (rng() % 10);
      list_type list;
      for (size_t j = 0; j < size; j++) {
        list.push_back(rng() % 20);
      }
      index.insert(i, list);
      lists.push_back(list);
      sets.emplace_back(list.begin(), list.end());
    }
  }

  bool disjoint(const set_type& a, const set_type& b) {
    counting_output_iterator out;
    return boost::set_intersection(a, b, out).count() == 0;
  }

  set_index<int, list_type> index;
  std::vector<list_type> lists;
  std::vector<set_type> sets;
};

BOOST_FIXTURE_TEST_CASE(test_superset, fixture) {
  for (size_t i = 0; i < n; ++i) {
    std::vector<int> results;
    index.find_supersets(lists[i], std::back_inserter(results));

    // check the answers are sound
    for (size_t j = 0; j < results.size(); ++j) {
      BOOST_CHECK(boost::includes(sets[results[j]], sets[i]));
    }

    // check the answers are complete
    size_t num_supersets = 0;
    for (int j = 0; j < n; ++j) {
      if (boost::includes(sets[j], sets[i])) {
        ++num_supersets;
      }
    }
    BOOST_CHECK_EQUAL(num_supersets, results.size());
  }
}

BOOST_FIXTURE_TEST_CASE(test_intersection, fixture) {
  for (int i = 0; i < n; ++i) {
    std::vector<int> results;
    index.find_intersecting_sets(lists[i], std::back_inserter(results));

    // Check the answers are sound.
    for (size_t j = 0; j < results.size(); ++j) {
      BOOST_CHECK(!disjoint(sets[results[j]], sets[i]));
    }

    // Check the answers are complete.
    size_t num_intersecting_sets = 0;
    for (size_t j = 0; j < n; j++) {
      if (!disjoint(sets[j], sets[i])) {
        ++num_intersecting_sets;
      }
    }
    BOOST_CHECK_EQUAL(num_intersecting_sets, results.size());
  }
}

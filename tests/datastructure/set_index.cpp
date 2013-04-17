#define BOOST_TEST_MODULE set_index
#include <boost/test/unit_test.hpp>

#include <vector>
#include <set>

#include <boost/random/mersenne_twister.hpp>

#include <sill/base/stl_util.hpp>
#include <sill/datastructure/set_index.hpp>

using std::size_t;
int n = 500;

struct fixture {
  typedef std::set<int> set_type;
  fixture() {
    // Make some random sets.
    for (int i = 0; i < n; i++) {
      set_type s;
      int size = rng() % 10;
      for (int j = 0; j < size; j++)
        s.insert(rng() % 20);
      sets.push_back(s);
      set_index.insert(s, i);
    }
  }

  boost::mt19937 rng;
  std::vector<set_type> sets;
  sill::set_index<set_type, int> set_index;
};

BOOST_FIXTURE_TEST_CASE(test_superset, fixture) {  
  for (int i = 0; i < n; ++i) {
    std::vector<int> results;
    set_type set = sets[i];
    set_index.find_supersets(set, std::back_inserter(results));

    // Check the answers are sound.
    for (unsigned int j = 0; j < results.size(); ++j) {
      BOOST_CHECK(sill::includes(sets[results[j]], set));
    }

    // Check the answers are complete.
    size_t num_supersets = 0;
    for (int j = 0; j < n; ++j) {
      if (sill::includes(sets[j], set))
        ++num_supersets;
    }
    BOOST_CHECK_EQUAL(num_supersets, results.size());
  }
}

BOOST_FIXTURE_TEST_CASE(test_intersection, fixture) {
  for (int i = 0; i < n; i++) {
    std::vector<int> results;
    set_type set = sets[i];
    set_index.find_intersecting_sets(set, std::back_inserter(results));

    // Check the answers are sound.
    for (unsigned int j = 0; j < results.size(); j++) {
      BOOST_CHECK(!sill::set_disjoint(sets[results[j]], set));
    }

    // Check the answers are complete.
    size_t num_intersecting_sets = 0;
    for (int j = 0; j < n; j++)
      if (!sill::set_disjoint(sets[j], set))
        num_intersecting_sets++;
    BOOST_CHECK_EQUAL(num_intersecting_sets, results.size());
  }
}

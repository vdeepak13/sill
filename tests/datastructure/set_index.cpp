#include <iostream>
#include <vector>
#include <cassert>
#include <set>

// #include <boost/graph/random.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <prl/base/stl_util.hpp>
#include <prl/datastructure/set_index.hpp>
using namespace prl;
int main(int argc, char** argv) {
  using std::size_t;
  boost::mt19937 rng;

  typedef std::set<int> set_type;
  prl::set_index<set_type, int> set_index;

  // Make some random sets.
  std::vector<set_type> sets;
  for (int i = 0; i < 1000; i++) {
    set_type s;
    int size = rng() % 10;
    for (int j = 0; j < size; j++)
      s.insert(rng() % 20);
    sets.push_back(s);
    set_index.insert(s, i);
  }

  // Run some queries.
  std::vector<int> results;

  // Issue superset queries and check the results.
  for (int i = 0; i < 1000; i++) {
    results.clear();
    set_type set = sets[i];
    set_index.find_supersets(set, std::back_inserter(results));
    // Check the answers are sound.
    for (unsigned int j = 0; j < results.size(); j++) {
      assert(includes(sets[results[j]], set));
    }
    // Check the answers are complete.
    size_t num_supersets = 0;
    for (unsigned int j = 0; j < 1000; j++) {
      if (includes(sets[j], set))
        num_supersets++;
    }
    assert(num_supersets == results.size());
  }

  // Issue intersection queries and check the results.
  for (int i = 0; i < 1000; i++) {
    results.clear();
    set_type set = sets[i];
    set_index.find_intersecting_sets(set, std::back_inserter(results));
    // Check the answers are sound.
    for (unsigned int j = 0; j < results.size(); j++) {
      //set_type intersecting_set = results[j].first;
      //if (intersecting_set.disjoint_from(set)) {
      //  std::cout << set << std::endl;
      //  std::cout << intersecting_set << std::endl;
      //}
      assert(!set_disjoint(sets[results[j]], set));
    }
    // Check the answers are complete.
    size_t num_intersecting_sets = 0;
    for (unsigned int j = 0; j < 1000; j++)
      if (!set_disjoint(sets[j], set))
        num_intersecting_sets++;
    assert(num_intersecting_sets == results.size());
  }

  // Return success.
  return EXIT_SUCCESS;
}

#define BOOST_TEST_MODULE assignment
#include <boost/test/unit_test.hpp>

#include <sill/base/finite_assignment.hpp>
#include <sill/base/finite_assignment_iterator.hpp>
#include <sill/base/vector_assignment.hpp>
#include <sill/base/universe.hpp>

#include <string>

#include <boost/array.hpp>
#include <boost/lexical_cast.hpp>

using namespace sill;

typedef std::map<int, std::string> int_string_map;
BOOST_TEST_DONT_PRINT_LOG_VALUE(int_string_map);
BOOST_TEST_DONT_PRINT_LOG_VALUE(std::set<int>);
BOOST_TEST_DONT_PRINT_LOG_VALUE(std::vector<std::string>);
BOOST_TEST_DONT_PRINT_LOG_VALUE(sill::finite_assignment_iterator);

std::map<int, std::string> make_map(int from, int to) {
  std::map<int, std::string> map;
  for (int i = from; i < to; ++i) {
    map[i] = boost::lexical_cast<std::string>(i);
  }
  return map;
}

std::set<int> make_set(int from, int to) {
  std::set<int> set;
  for (int i = from; i < to; ++i) {
    set.insert(i);
  }
  return set;
}

std::vector<std::string> make_values(int from, int to) {
  std::vector<std::string> vec;
  for (int i = from; i < to; ++i) {
    vec.push_back(boost::lexical_cast<std::string>(i));
  }
  return vec;
}

BOOST_AUTO_TEST_CASE(test_operations) {
  std::map<int, std::string> m1 = make_map(1, 5);
  std::map<int, std::string> m2 = make_map(3, 7);
  std::set<int> s1 = make_set(1, 5);
  std::set<int> s2 = make_set(3, 7);
  std::set<int> s12 = make_set(3, 5);
  std::vector<int> v12; v12.push_back(3); v12.push_back(4);
  
  BOOST_CHECK_EQUAL(safe_get(m1, 3), "3");
  BOOST_CHECK_EQUAL(*get_ptr(m1, 3), "3");
  BOOST_CHECK_EQUAL(get_ptr(m1, 10), static_cast<std::string*>(NULL));
  BOOST_CHECK_EQUAL(map_union(m1, m2), make_map(1, 7));
  BOOST_CHECK_EQUAL(map_intersect(m1, m2), make_map(3, 5));
  BOOST_CHECK_EQUAL(map_intersect(m1, s2), make_map(3, 5));
  BOOST_CHECK_EQUAL(map_difference(m1, m2), make_map(1, 3));
  BOOST_CHECK_EQUAL(keys(m2), s2);
  
  std::vector<std::string> v1(values(m1).first, values(m1).second);
  BOOST_CHECK_EQUAL(v1, make_values(1, 5));
  BOOST_CHECK_EQUAL(values(m1, s12), make_values(3, 5));
  BOOST_CHECK_EQUAL(values(m1, v12), make_values(3, 5));
  
  std::map<int,int> id = make_identity_map(s12);
  BOOST_CHECK_EQUAL(id.size(), 2);
  BOOST_CHECK_EQUAL(id[3], 3);
  BOOST_CHECK_EQUAL(id[4], 4);

  std::map<int, std::string> m1_copy = rekey(m1, make_identity_map(s1));
  BOOST_CHECK_EQUAL(m1, m1_copy);
  
  std::map<std::string, std::string> value_map;
  std::map<int, std::string> m1_mapped;
  for (int i = 1; i < 5; ++i) {
    std::string istr = boost::lexical_cast<std::string>(i);
    value_map[istr] = istr + "_mapped";
    m1_mapped[i] = istr + "_mapped";
  }
  BOOST_CHECK_EQUAL(remap(m1, value_map), m1_mapped);
}

struct fixture {
  fixture()
    : v(u.new_finite_variable(2)),
      x(u.new_finite_variable(3)),
      y(u.new_vector_variable(2)) { }
  
  universe u;
  finite_variable* v;
  finite_variable* x;
  vector_variable* y;
};

BOOST_FIXTURE_TEST_CASE(test_assignments, fixture) {
  // Create an assignment.
  finite_assignment fa;
  vector_assignment va;
  fa[x] = 2;
  // va[y] = vector_variable::value_type(2); 
  // warning: this only initializes the dimensions, not the content
  va[y] = zeros(2);

  BOOST_CHECK_EQUAL(fa[x], 2);
  BOOST_CHECK(equal(va[y], vec(zeros(2))));
}

BOOST_FIXTURE_TEST_CASE(test_finite_assignment_iterator, fixture) {
  // Test the assignment iterator
  boost::array<finite_variable*, 2> d = {{ v, x }};
  finite_assignment_iterator it(d), end;
  finite_assignment fa;
  for (size_t i = 0; i < 3; ++i) {
    fa[x] = i;
    for (size_t j = 0; j < 2; ++j) {
      fa[v] = j;
      BOOST_CHECK_NE(it, end);
      BOOST_CHECK_EQUAL(*it++, fa);
    }
  }
  BOOST_CHECK_EQUAL(it, end);

  // Test the empty iterator
  boost::array<finite_variable*, 0> empty;
  it = finite_assignment_iterator(empty);
  fa.clear();
  BOOST_CHECK_EQUAL(*it, fa);
  BOOST_CHECK_EQUAL(++it, end);
}

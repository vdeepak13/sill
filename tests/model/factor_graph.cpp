#define BOOST_TEST_MODULE factor_graph
#include <boost/test/unit_test.hpp>

#include <boost/array.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <sill/stl_io.hpp>
#include <sill/base/universe.hpp>
#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/random/uniform_factor_generator.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/model/factor_graph.hpp>
#include <sill/range/algorithm.hpp>
#include <sill/serialization/serialize.hpp>

#include "predicates.hpp"

#include <sill/macros_def.hpp>

using namespace sill;

template class factor_graph<table_factor>;
template class factor_graph<canonical_gaussian>;

struct fixture {
  typedef factor_graph<table_factor> model_type;
  
  fixture() 
    : nvars(10) {
    // Random number generator
    boost::mt19937 rng;
    uniform_factor_generator gen;

    // Create some variables
    x.resize(nvars);
    for(size_t i = 0; i < nvars; ++i) {
      x[i] = u.new_finite_variable("Variable: " + to_string(i), 2);
    }

    // Create some unary factors
    for(size_t i = 0; i < nvars; ++i) {
      finite_domain arguments = make_domain(x[i]);
      fg.add_factor(gen(arguments, rng));
    }
    
    // For every two variables in a chain create a factor
    for(size_t i = 0; i < x.size() - 1; ++i) {
      finite_domain arguments = make_domain(x[i], x[i+1]);
      fg.add_factor(gen(arguments, rng));
    }
  }

  universe u;
  size_t nvars;
  std::vector<finite_variable*> x;
  model_type fg;
};

struct domain_less {
  template <typename Set>
  bool operator()(const Set& a, const Set& b) {
    return sill::lexicographical_compare(a, b);
  }
};

BOOST_FIXTURE_TEST_CASE(test_structure, fixture) {
  for (size_t i = 0; i < nvars; ++i) {
    std::vector<finite_domain> args1;
    args1.push_back(make_domain(x[i]));
    if (i > 0) {
      args1.push_back(make_domain(x[i-1], x[i]));
    }
    if (i < nvars - 1) {
      args1.push_back(make_domain(x[i], x[i+1]));
    }

    std::vector<finite_domain> args2;
    foreach(size_t id, fg.neighbors(x[i])) {
      args2.push_back(fg.cluster(id));
    }
    
    sill::sort(args1, domain_less());
    sill::sort(args2, domain_less());
    BOOST_CHECK_EQUAL(args1, args2);
  }
  BOOST_CHECK_EQUAL(fg.num_variables(), nvars);
  BOOST_CHECK_EQUAL(fg.num_factors(), 2*nvars - 1);
  BOOST_CHECK_EQUAL(fg.num_edges(), 2*(nvars-1) + nvars);
}

BOOST_FIXTURE_TEST_CASE(test_simplify, fixture) {
  fg.simplify();
  for (size_t i = 0; i < nvars; ++i) {
    std::vector<finite_domain> args1;
    if (i > 0) {
      args1.push_back(make_domain(x[i-1], x[i]));
    }
    if (i < nvars - 1) {
      args1.push_back(make_domain(x[i], x[i+1]));
    }

    std::vector<finite_domain> args2;
    foreach(size_t id, fg.neighbors(x[i])) {
      args2.push_back(fg.cluster(id));
    }
    
    sill::sort(args1, domain_less());
    sill::sort(args2, domain_less());
    BOOST_CHECK_EQUAL(args1, args2);
  }
  BOOST_CHECK_EQUAL(fg.num_variables(), nvars);
  BOOST_CHECK_EQUAL(fg.num_factors(), nvars - 1);
  BOOST_CHECK_EQUAL(fg.num_edges(), 2*(nvars-1));
}

BOOST_FIXTURE_TEST_CASE(test_serialization, fixture) {
  BOOST_CHECK(serialize_deserialize(fg, u));
}

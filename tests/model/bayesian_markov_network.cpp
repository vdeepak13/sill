#define BOOST_TEST_MODULE bayesian_markov_network
#include <boost/test/unit_test.hpp>

#include <boost/array.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/datastructure/dense_table.hpp>
#include <sill/model/bayesian_network.hpp>
#include <sill/model/markov_network.hpp>

#include "predicates.hpp"

using namespace sill;
using boost::array;

template class bayesian_network<table_factor>;
template class markov_network<table_factor>;

template class bayesian_network<canonical_gaussian>;
template class markov_network<canonical_gaussian>;

template class bayesian_network<moment_gaussian>;

// TODO: check bayesian network conditioning

struct fixture {
  fixture()
    : x(u.new_finite_variables(5, 2)) {
    /* Create factors for the Bayesian network with this structure:
     * 0, 1 (no parents)
     * 1 --> 2
     * 1,2 --> 3
     * 0,3 --> 4
     */

    finite_var_vector a0 = make_vector(x[0]);
    array<double, 2> v0 = {{.3, .7}};
    
    finite_var_vector a1 = make_vector(x[1]);
    array<double, 2> v1 = {{.5, .5}};
    
    finite_var_vector a12 = make_vector(x[1], x[2]);
    array<double, 4> v12 = {{.8, .2, .2, .8}};
    
    finite_var_vector a123 = make_vector(x[1], x[2], x[3]);
    array<double, 8> v123 = {{.1, .1, .3, .5, .9, .9, .7, .5}};
    
    finite_var_vector a034 = make_vector(x[0], x[3], x[4]);
    array<double, 8> v034 = {{.6, .1, .2, .1, .4, .9, .8, .9}};

    f0 = make_dense_table_factor(a0, v0);
    f1 = make_dense_table_factor(a1, v1);
    f12 = make_dense_table_factor(a12, v12);
    f123 = make_dense_table_factor(a123, v123);
    f034 = make_dense_table_factor(a034, v034);

    bn.add_factor(x[0], f0);
    bn.add_factor(x[1], f1);
    bn.add_factor(x[2], f12);
    bn.add_factor(x[3], f123);
    bn.add_factor(x[4], f034);

    mn = bayes2markov_network(bn);
  }

  universe u;
  finite_var_vector x;
  bayesian_network<table_factor> bn;
  markov_network<table_factor> mn;
  table_factor f0, f1, f12, f123, f034;
};

BOOST_FIXTURE_TEST_CASE(test_conversion, fixture) {
  BOOST_CHECK(model_equal_factors(bn, mn));
  BOOST_CHECK(model_close_log_likelihoods(bn, mn, 1e-6));
}

BOOST_FIXTURE_TEST_CASE(test_serialization, fixture) {
  BOOST_CHECK(serialize_deserialize(bn, u));
  BOOST_CHECK(serialize_deserialize(mn, u));
}

BOOST_FIXTURE_TEST_CASE(test_conditioning, fixture) {
  finite_assignment a;
  a[x[0]] = 0;
  a[x[2]] = 1;
  mn.condition(a);

  markov_network<table_factor> mn2;
  //mn2.add_factor(f0.restrict(a));
  mn2.add_factor(f1.restrict(a));
  mn2.add_factor(f12.restrict(a));
  mn2.add_factor(f123.restrict(a));
  mn2.add_factor(f034.restrict(a));

  BOOST_CHECK(model_close_log_likelihoods(mn, mn2, 1e-6));
}

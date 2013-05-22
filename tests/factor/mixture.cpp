#define BOOST_TEST_MODULE mixture
#include <boost/test/unit_test.hpp>

#include <iostream>

#include <sill/factor/mixture.hpp>
#include <sill/factor/moment_gaussian.hpp>

#include "predicates.hpp"

using namespace sill;

struct fixture {
  fixture()
    : v(u.new_vector_variables(2, 1)),
      vars(v.begin(), v.end()),
      mix(2, vars) {
    mix[0] = moment_gaussian(v, zeros(2), eye(2,2));
    mix[1] = moment_gaussian(v, ones(2), "2 1; 1 3");
  }
  universe u;
  vector_var_vector v;
  vector_domain vars;
  mixture_gaussian mix;
};

// test the construtor
BOOST_FIXTURE_TEST_CASE(test_constructor, fixture) {
  BOOST_CHECK_EQUAL(mix.size(), 2);
  BOOST_CHECK_EQUAL(mix.arguments(), vars);
  BOOST_CHECK_CLOSE(mix.norm_constant(), 2.0, 1e-10);
  BOOST_CHECK_EQUAL(mix[0], moment_gaussian(v, zeros(2), eye(2,2)));
  BOOST_CHECK_EQUAL(mix[1], moment_gaussian(v, ones(2), "2 1; 1 3"));
}

// test various operations (TODO: add more)
BOOST_FIXTURE_TEST_CASE(test_operations, fixture) {
  mix.add_parameters(mix, 0.5);
  BOOST_CHECK_EQUAL(mix.size(), 2);
  BOOST_CHECK_CLOSE(mix.norm_constant(), 3.0, 1e-10);
  moment_gaussian m0(v, zeros(2), 1.5*eye(2,2), 1.5);
  moment_gaussian m1(v, "1.5 1.5", "3 1.5; 1.5 4.5", 1.5);
  BOOST_CHECK(are_close(mix[0], m0, 1e-10));
  BOOST_CHECK(are_close(mix[1], m1, 1e-10));
  
  mix.normalize();
  BOOST_CHECK_EQUAL(mix.size(), 2);
  BOOST_CHECK_EQUAL(mix.arguments(), vars);
  BOOST_CHECK_CLOSE(mix.norm_constant(), 1.0, 1e-10);
}

// serialize and deserialize
BOOST_FIXTURE_TEST_CASE(test_serialization, fixture) {
  BOOST_CHECK(serialize_deserialize(mix, u));
}

// KL projection
BOOST_AUTO_TEST_CASE(test_projection) {
  universe u;
  vector_domain args;
  args.insert(u.new_vector_variable("x", 2));

  mixture_gaussian mix(2, args);
  mix[0] = moment_gaussian(make_vector(args), "1 2", eye(2, 2));
  mix[1] = moment_gaussian(make_vector(args), "-1 -2", eye(2, 2));

  moment_gaussian projection_true(make_vector(args), "0 0", "2 2; 2 5", 
                                  logarithmic<double>(0.693147, log_tag()));
  BOOST_CHECK(are_close(project(mix), projection_true, 1e-5));
}

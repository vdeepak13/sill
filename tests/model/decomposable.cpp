#define BOOST_TEST_MODULE decomposable
#include <boost/test/unit_test.hpp>

#include <boost/random/mersenne_twister.hpp>

#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/model/decomposable.hpp>
#include <sill/model/random.hpp>

#include "basic_fixture.hpp"
#include "predicates.hpp"

template class decomposable<table_factor>;
template class decomposable<canonical_gaussian>;

BOOST_FIXTURE_TEST_CASE(test_marginal, basic_fixture) {
  decomposable<table_factor> model;
  model *= factors;

  finite_domain dom = make_domain(lvfailure,history,cvp,pcwp,hypovolemia);
  table_factor marginal = model.marginal(dom);
  BOOST_CHECK_CLOSE(marginal.entropy(), 4.27667, 1e-3);
  BOOST_CHECK_EQUAL(marginal.arguments(), dom);

  decomposable<table_factor> marginal_model;
  model.marginal(dom, marginal_model);
  BOOST_CHECK_EQUAL(marginal_model.arguments(), dom);
  BOOST_CHECK_CLOSE(marginal_model.entropy(), 4.27667, 1e-3);
}

BOOST_FIXTURE_TEST_CASE(test_copy, basic_fixture) {
  decomposable<table_factor> model;
  model *= factors;
  model.check_validity();
  // TODO: turn this into a bool function

  decomposable<table_factor> model2(model);
  model2.check_validity();
  BOOST_CHECK_EQUAL(model, model2);
}

BOOST_FIXTURE_TEST_CASE(test_serialization, basic_fixture) {
  decomposable<table_factor> model;
  model *= factors;
  BOOST_CHECK(serialize_deserialize(model, u));
}

BOOST_AUTO_TEST_CASE(test_mpa) {
  universe u;
  decomposable<table_factor> model3;
  bayesian_network<table_factor> model3_bn;
  boost::mt11213b rng(4350198);
  random_HMM(model3_bn, rng, u, 10, 4, 4, 0.5, 0.5);
  model3 *= model3_bn.factors();

  finite_assignment mpa = model3.max_prob_assignment();
  BOOST_CHECK_CLOSE(model3.log_likelihood(mpa), -9.74229, 1e-2);
}

BOOST_AUTO_TEST_CASE(test_sampling) {
  // Dataset parameters
  size_t nsamples = 500;
  size_t n = 30; // length of width-2 chain decomposable model
  boost::mt11213b rng;

  // Create a model to sample from
  universe u;
  bayesian_network<table_factor> bn;
  random_HMM(bn, rng, u, n, 2, 2);
  decomposable<table_factor> model(bn.factors());

  // Test conditioning and computing log likelihoods.
  finite_domain half_vars1(model.arguments());
  finite_domain half_vars2;
  for (size_t i = 0; i < n / 2; ++i) {
    assert(!half_vars1.empty());
    finite_variable* v = *(half_vars1.begin());
    half_vars1.erase(v);
    half_vars2.insert(v);
  }
  decomposable<table_factor> half_vars1_model;
  model.marginal(half_vars1, half_vars1_model);

  // Sample
  double true_entropy = model.entropy();
  double cross_entropy = 0;
  double ll_half_vars1 = 0;
  double ll_half_vars2_given_1 = 0;
  for (size_t i = 0; i < nsamples; ++i) {
    finite_assignment a(model.sample(rng));
    cross_entropy -= model.log_likelihood(a);
    decomposable<table_factor> conditioned_model(model);
    finite_assignment a_half_vars1(map_intersect(a, half_vars1));
    conditioned_model.condition(a_half_vars1);
    ll_half_vars1 += half_vars1_model.log_likelihood(a);
    ll_half_vars2_given_1 += conditioned_model.log_likelihood(a);
  }

  double estimate = cross_entropy / nsamples;
  double estimate1 = -(ll_half_vars1 + ll_half_vars2_given_1) / nsamples;
  BOOST_CHECK_CLOSE(true_entropy, estimate, 1.0);
  BOOST_CHECK_CLOSE(true_entropy, estimate1, 1.0);
}

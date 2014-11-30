#define BOOST_TEST_MODULE table_factor
#include <boost/test/unit_test.hpp>

#include <iostream>

#include <boost/random/mersenne_twister.hpp>

#include <sill/factor/table_factor.hpp>
#include <sill/factor/random/uniform_factor_generator.hpp>
#include <sill/learning/dataset_old/record_conversions.hpp>

#include "predicates.hpp"

using namespace sill;

struct fixture {
  fixture()
    : vars(u.new_finite_variables(3, 2)),
      f(gen(make_domain(vars), rng)) { }

  universe u;
  boost::mt19937 rng; 
  uniform_factor_generator gen;
  finite_var_vector vars;
  table_factor f;
};

BOOST_FIXTURE_TEST_CASE(test_product, fixture) {
  finite_var_vector doma = make_vector(vars[0], vars[1]);
  finite_var_vector domb = make_vector(vars[1], vars[2]);

  boost::array<double, 4> vala = {{ 1, 2, 3, 4 }};
  boost::array<double, 4> valb = {{ 1, 0.5, 2, 3 }};

  table_factor fa = make_dense_table_factor(doma, vala);
  table_factor fb = make_dense_table_factor(domb, valb);
  finite_assignment assign;
  
  std::vector<double> valab(8, 0.0);
  for (size_t i = 0; i < 2; ++i) {
    assign[vars[0]] = i;
    for (size_t j = 0; j < 2; ++j) {
      assign[vars[1]] = j;
      for (size_t k = 0; k < 2; ++k) {
        assign[vars[2]] = k;
        valab[i + 2*j + 4*k] = fa(assign) * fb(assign);
      }
    }
  }
  
  table_factor fab(vars, valab);
  BOOST_CHECK_EQUAL(fa * fb, fab);
}

BOOST_FIXTURE_TEST_CASE(test_marginal, fixture) {
  finite_assignment a;
  std::vector<double> sum(4, 0.0);
  for (size_t i = 0; i < 2; ++i) {
    a[vars[0]] = i;
    for (size_t j = 0; j < 2; ++j) {
      a[vars[1]] = j;
      for (size_t k = 0; k < 2; ++k) {
        a[vars[2]] = k;
        sum[i + 2 * j] += f(a);
      }
    }
  }
  table_factor true_marginal(make_vector(vars[0], vars[1]), sum);
  finite_domain dom = make_domain(vars[0], vars[1]);
  BOOST_CHECK_EQUAL(f.marginal(dom), true_marginal);
}

BOOST_FIXTURE_TEST_CASE(test_restrict, fixture) {
  finite_assignment fa;
  fa[vars[0]] = 1;
  fa[vars[1]] = 0;
  finite_var_vector fa_vars(vars.begin(), vars.begin() + 2);
  std::vector<size_t> fr_data;
  finite_assignment2vector(fa, fa_vars, fr_data);
  finite_record_old fr(fa_vars);
  fr.set_finite_val(fr_data);

  table_factor f1(f);
  table_factor new_f1a;
  f1.restrict(fa, new_f1a);
  table_factor new_f1b;
  f1.restrict(fr, new_f1b);
  BOOST_CHECK_EQUAL(new_f1a, new_f1b);

  f1.restrict(fa, make_domain(vars[1]), new_f1a);
  f1.restrict(fr, make_domain(vars[1]), new_f1b);
  BOOST_CHECK_EQUAL(new_f1a, new_f1b);
}

BOOST_FIXTURE_TEST_CASE(test_rolling, fixture) {
  // do not seed with time for reproducibility
  finite_variable* f_unrolled_v = NULL;
  table_factor f_unrolled;
  boost::tie(f_unrolled_v, f_unrolled) = f.unroll(u);
  table_factor f_rolled_again = f_unrolled.roll_up(f.arg_vector());
  BOOST_CHECK_EQUAL(f, f_rolled_again); 
}

BOOST_AUTO_TEST_CASE(test_pow) {
  universe u;
  finite_var_vector vars = u.new_finite_variables(2, 2);
  std::vector<double> values;
  std::vector<double> values_pow;
  for (size_t i = 0; i < 4; ++i) {
    values.push_back(i);
    values_pow.push_back(i * i);
  }
  
  table_factor f(vars, values);
  table_factor f_pow(vars, values_pow);
  BOOST_CHECK(are_close(pow(f, 2), f_pow, 1e-10));
}

BOOST_AUTO_TEST_CASE(test_reorder) {
  universe u;
  finite_variable* a = u.new_finite_variable("a", 2);
  finite_variable* b = u.new_finite_variable("b", 3);
  
  finite_var_vector ab = make_vector(a, b);
  finite_var_vector ba = make_vector(b, a);

  boost::array<double, 6> ab_vals = {0.5, 1, 1.5, 2, 2.5, 3};
  boost::array<double, 6> ba_vals = {0.5, 1.5, 2.5, 1, 2, 3};
  
  table_factor ab_f = make_dense_table_factor(ab, ab_vals);
  table_factor ba_f = make_dense_table_factor(ba, ba_vals);
  
  BOOST_CHECK(ab_f.reorder(ab) == ab_f);
  BOOST_CHECK(ab_f.reorder(ba) == ba_f);
}

BOOST_AUTO_TEST_CASE(test_sampling) {
  // dataset parameters
  size_t nsamples = 100000;
  size_t nvars = 2;
  size_t arity = 4;

  universe u;
  finite_domain vars;
  for (size_t i = 0; i < nvars; ++i)
    vars.insert(u.new_finite_variable(arity));
  boost::mt11213b rng;

  // Create a model to sample from.
  table_factor f = uniform_factor_generator()(vars, rng);
  f.normalize();

  // Test log likelihood.
  double true_entropy = f.entropy();
  double cross_entropy = 0;
  for (size_t i = 0; i < nsamples; ++i) {
    finite_assignment a(f.sample(rng));
    cross_entropy -= f.logv(a);
  }
  BOOST_CHECK_CLOSE(true_entropy, cross_entropy / nsamples, 1.0 /* percent */);

  // TODO: test conditioning and computing log likelihoods
}

BOOST_AUTO_TEST_CASE(test_marginal_sampler_mle) {
  // dataset parameters
  size_t nsamples = 100000;
  size_t nvars = 2;
  size_t arity = 4;

  universe u;
  finite_domain vars;
  for (size_t i = 0; i < nvars; ++i)
    vars.insert(u.new_finite_variable(arity));
  boost::mt11213b rng;

  // Create a model to sample from
  table_factor f = uniform_factor_generator()(vars, rng);
  f.normalize();

  // Test log likelihood
  double true_entropy = f.entropy();
  double cross_entropy = 0;
  std::vector<size_t> sample;
  factor_sampler<table_factor> sampler(f);
  factor_mle_incremental<table_factor> mle(f.arg_vector());
  for (size_t i = 0; i < nsamples; ++i) {
    sampler(sample, rng);
    cross_entropy -= std::log(f(sample));
    mle.process(sample, 1.0);
  }
  BOOST_CHECK_CLOSE(true_entropy, cross_entropy / nsamples, 1.0 /* percent */);
  BOOST_CHECK_SMALL(f.relative_entropy(mle.estimate()), 1e-2);
  BOOST_CHECK_CLOSE(mle.weight(), nsamples, 1e-2 /* percent */);
}

BOOST_AUTO_TEST_CASE(test_conditional_sampler_mle) {
  // dataset parameters
  size_t nsamples = 100000;
  size_t arity = 4;

  universe u;
  finite_variable* head = u.new_finite_variable(arity);
  finite_variable* tail = u.new_finite_variable(arity);

  // Create a model to sample from
  boost::mt11213b rng;
  uniform_factor_generator gen;
  table_factor f = gen(make_domain(head), make_domain(tail), rng);

  // For each assignment to tail, draw a number of samples and compare to f
  factor_sampler<table_factor> sampler(f, make_vector(head));
  factor_mle_incremental<table_factor> mle(make_vector(head), make_vector(tail));
  std::vector<size_t> sample;
  for (size_t val = 0; val < arity; ++val) {
    std::vector<size_t> tail_index(1, val);
    std::vector<size_t> index(2, val);
    table_factor g(make_domain(head), 0);
    for (size_t i = 0; i < nsamples; ++i) {
      sampler(sample, tail_index, rng);
      ++g(sample);
      index[1] = sample[0];
      mle.process(index, 1.0);
    }
    double diff = norm_1(f.restrict(make_assignment(tail, val)), g / nsamples);
    BOOST_CHECK_SMALL(diff, 0.01);
  }
  BOOST_CHECK_SMALL(norm_1(f, mle.estimate()), 5e-2);
  BOOST_CHECK_CLOSE(mle.weight(), nsamples * arity, 1e-2 /* percent */);
}

BOOST_AUTO_TEST_CASE(test_comparison) {
  universe u;

  finite_variable* x = u.new_finite_variable(2);
  finite_variable* y = u.new_finite_variable(2);
  finite_variable* z = u.new_finite_variable(2);
  finite_var_vector xy = make_vector(x, y);
  finite_var_vector yx = make_vector(y, x);
  finite_var_vector xz = make_vector(x, z);

  boost::array<double, 4> v1 = {{ 1, 2, 3, 4 }};
  boost::array<double, 4> v2 = {{ 1, 3, 2, 4 }};
  boost::array<double, 4> v3 = {{ 1, 2, 3, 0 }};

  table_factor f1 = make_dense_table_factor(xy, v1);
  table_factor f2 = make_dense_table_factor(yx, v2);
  table_factor f3 = make_dense_table_factor(xy, v3);
  table_factor f4 = make_dense_table_factor(xz, v1);
  table_factor f5 = f1;

  BOOST_CHECK_EQUAL(f1, f2);
  BOOST_CHECK_NE(f1, f3);
  BOOST_CHECK_NE(f2, f3);
  BOOST_CHECK_NE(f3, f4);
  BOOST_CHECK_EQUAL(f1, f5);
  BOOST_CHECK_EQUAL(f2, f5);
//   BOOST_CHECK_LT(f3, f2);
//   BOOST_CHECK_LT(f3, f4);
//   BOOST_CHECK_LT(f2, f4);
}

BOOST_FIXTURE_TEST_CASE(test_serialization, fixture) {
  BOOST_CHECK(serialize_deserialize(f, u));
}

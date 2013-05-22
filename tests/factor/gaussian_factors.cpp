#define BOOST_TEST_MODULE gaussian_factors
#include <boost/test/unit_test.hpp>

#include <iostream>

#include <boost/random/lagged_fibonacci.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/moment_gaussian.hpp>
#include <sill/factor/operations.hpp>

#include "predicates.hpp"

using namespace sill;

struct fixture {
  fixture()
    : x(u.new_vector_variable("x", 1)),
      y(u.new_vector_variable("y", 1)),
      z(u.new_vector_variable("z", 1)),
      q(u.new_vector_variable("q", 1)),

      dom_x(make_vector(x)),
      dom_y(make_vector(y)),
      dom_xy(make_vector(x, y)),
      dom_yx(make_vector(y, x)),
      dom_yz(make_vector(y, z)),
      dom_zq(make_vector(z, q)),
      dom_xyz(make_vector(x, y, z)),

      ma (mat_2x2(2.0, 1.0, 1.0, 3.0)),
      ma_(mat_2x2(3.0, 1.0, 1.0, 2.0)),
      mb (mat_2x2(3.0, 2.0, 2.0, 4.0)),
      mc (mat_2x2(2.0, 1.0, 1.0, 5.0)),
      
      va (vec_2(1.0, 2.0)),
      va_(vec_2(2.0, 1.0)),
      vb (vec_2(3.0, 4.0)) { }
  
  universe u;
  vector_variable* x;
  vector_variable* y;
  vector_variable* z; 
  vector_variable* q;

  vector_var_vector dom_x, dom_y, dom_xy, dom_yx, dom_yz, dom_zq, dom_xyz;

  mat ma, ma_, mb, mc;
  vec va, va_, vb;
};

struct fixture_cg : public fixture {
  fixture_cg()
    : f_xy_a_a(dom_xy, ma, va),
      f_yx_a_a(dom_yx, ma_, va_),
      f_yz_b_b(dom_yz, mb, vb),
      f_xy_b_b(dom_xy, mb, vb),
      f_xy_a_b(dom_xy, ma, vb) { }
  
  canonical_gaussian f_xy_a_a;
  canonical_gaussian f_yx_a_a;
  canonical_gaussian f_yz_b_b;
  canonical_gaussian f_xy_b_b;
  canonical_gaussian f_xy_a_b;
};

struct fixture_mg : public fixture {
  fixture_mg()
    : f_xy_a_c(dom_xy, va, mc) { }

  moment_gaussian f_xy_a_c;
};

BOOST_FIXTURE_TEST_CASE(cg_comparisons, fixture_cg) {
  BOOST_CHECK_EQUAL(f_xy_a_a, f_xy_a_a);
  BOOST_CHECK_EQUAL(f_xy_a_a, f_yx_a_a);
  BOOST_CHECK_EQUAL(f_xy_a_a < f_yz_b_b, x < y);
  BOOST_CHECK_NE(f_xy_a_a, f_yz_b_b);
  BOOST_CHECK_NE(f_xy_a_a, f_xy_b_b);
  BOOST_CHECK_LT(f_xy_a_a, f_xy_b_b);
  BOOST_CHECK_LT(f_xy_a_b, f_xy_b_b);
}

BOOST_FIXTURE_TEST_CASE(cg_opertions, fixture_cg) {
  canonical_gaussian f_xyz(dom_xyz, "2 1 0; 1 6 2; 0 2 4", "1 5 4");
  BOOST_CHECK_EQUAL(f_xy_a_a * f_yz_b_b, f_xyz);

  vector_assignment assign0;
  assign0[y] = zeros(1);
  assign0[z] = zeros(1);
  BOOST_CHECK_EQUAL(f_xyz.restrict(assign0), canonical_gaussian(dom_x, "2", "1", 0.0));

  vector_assignment assign1;
  assign1[y] = ones(1);
  assign1[z] = ones(1);
  BOOST_CHECK_EQUAL(f_xyz.restrict(assign1), canonical_gaussian(dom_x, "2", "0", 2.0));

  canonical_gaussian marginal = f_xyz.marginal(make_domain(x));
  BOOST_CHECK(are_close(marginal, canonical_gaussian(dom_x, "1.8", "0.4", 3.24001), 1e-4));

  vector_var_map vm;
  vm[x] = z;
  vm[y] = q;
  BOOST_CHECK_EQUAL(f_xy_a_a.subst_args(vm), canonical_gaussian(dom_zq, ma, va));
}

BOOST_FIXTURE_TEST_CASE(cg_serialization, fixture_cg) {
  BOOST_CHECK(serialize_deserialize(f_xy_a_a, u));
  BOOST_CHECK(serialize_deserialize(f_yz_b_b, u));
}

BOOST_FIXTURE_TEST_CASE(mg_operations, fixture_mg) {
  // marginal moment Gaussian
  vec val = vec_2(0.5, 0.5);
  BOOST_CHECK_CLOSE(log(f_xy_a_c(val)), -3.1726, 1e-1);
  
  vector_assignment a; 
  a[y] = ones(1);
  moment_gaussian restricted_true(dom_x, "0.8", "1.8",
                                  logarithmic<double>(-1.8236574894, log_tag()));
  BOOST_CHECK(are_close(f_xy_a_c.restrict(a), restricted_true, 1e-6));

  // conditional moment Gaussians
  moment_gaussian mg(dom_xy, va, mc, dom_zq, mb);
  a.clear();
  a[x] = vec("0.5");
  a[y] = vec("0.5");
  a[z] = ones(1);
  a[q] = ones(1);
  moment_gaussian restricted = mg.restrict(a);
  BOOST_CHECK(restricted.arguments().empty());
  BOOST_CHECK_CLOSE(log(restricted.norm_constant()), -13.0059, 1e-3);

  a.clear();
  a[z] = ones(1);
  a[q] = ones(1);
  restricted_true = moment_gaussian(dom_xy, vec_2(6.0, 8.0), mc);
  BOOST_CHECK_EQUAL(mg.restrict(a), restricted_true);
  BOOST_CHECK_CLOSE(log(mg.restrict(a)(val)), -13.0059, 1e-3);
}

// test sampling, conditioning, and restricting.
BOOST_FIXTURE_TEST_CASE(mg_sampling, fixture_mg) {
  moment_gaussian mg_xy(dom_xy, va, mc); // [1 2], [1 2; 2 5]
  moment_gaussian mg_x_given_y(mg_xy.conditional(make_domain(y)));
  moment_gaussian mg_y(mg_xy.marginal(make_domain(y)));
  boost::mt11213b rng(2359817);
  double mg_xy_ll = 0.0;
  double mg_x_given_y_ll = 0.0;
  double mg_y_ll = 0.0;
  size_t nsamples = 100;
  for (size_t i = 0; i < nsamples; ++i) {
    vector_assignment va(mg_xy.sample(rng));
    mg_xy_ll += mg_xy.logv(va);
    vector_assignment va_y;
    va_y[y] = va[y];
    mg_x_given_y_ll += mg_x_given_y.restrict(va_y).logv(va);
    mg_y_ll += mg_y.logv(va);
  }
  mg_xy_ll /= nsamples;
  mg_x_given_y_ll /= nsamples;
  mg_y_ll /= nsamples;

  BOOST_CHECK_CLOSE(mg_xy_ll, -4.04523, 1e-3);        // E[log P(x,y)]
  BOOST_CHECK_CLOSE(mg_x_given_y_ll, -1.74833, 1e-3); // E[log P(x|y)]
  BOOST_CHECK_CLOSE(mg_y_ll, -2.2969, 1e-3);          // E[log P(y)]
  BOOST_CHECK_CLOSE(mg_x_given_y_ll + mg_y_ll, -4.04523, 1e-3);
                                      // E[log P(x|y)] + E[log P(y)]
}

BOOST_FIXTURE_TEST_CASE(mg_serialization, fixture_mg) {
  BOOST_CHECK(serialize_deserialize(f_xy_a_c, u));
}

BOOST_FIXTURE_TEST_CASE(conversions, fixture) {
  moment_gaussian mg(dom_x, zeros(1), eye(1,1), dom_y, ones(1,1));
  canonical_gaussian cg(dom_xy, mat("1 -1; -1 1"), zeros(2), -0.918939);
  BOOST_CHECK(are_close(canonical_gaussian(mg), cg, 1e-5));
}

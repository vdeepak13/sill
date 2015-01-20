#define BOOST_TEST_MODULE canonical_table
#include <boost/test/unit_test.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/canonical_table.hpp>

#include <boost/range/algorithm.hpp>

#include "predicates.hpp"

namespace sill {
  template class canonical_table<double>;
  template class canonical_table<float>;
}

using namespace sill;

typedef canonical_table<double> ct_type;
typedef canonical_table<double>::param_type param_type;
typedef logarithmic<double> logd;

BOOST_AUTO_TEST_CASE(test_constructors) {
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 3);

  ct_type a;
  BOOST_CHECK(a.empty());
  BOOST_CHECK(a.arguments().empty());
  BOOST_CHECK(a.arg_vector().empty());

  ct_type b({x, y});
  BOOST_CHECK(table_properties(b, {x, y}));

  ct_type c(2.0);
  BOOST_CHECK(table_properties(c, {}));
  BOOST_CHECK_CLOSE(c[0], std::log(2.0), 1e-8);
  
  ct_type d(make_vector(x), logd(3.0));
  BOOST_CHECK(table_properties(d, {x}));
  BOOST_CHECK_CLOSE(d[0], std::log(3.0), 1e-8);
  BOOST_CHECK_CLOSE(d[1], std::log(3.0), 1e-8);

  ct_type e(make_domain(x), logd(4.0));
  BOOST_CHECK(table_properties(e, {x}));
  BOOST_CHECK_CLOSE(e[0], std::log(4.0), 1e-8);
  BOOST_CHECK_CLOSE(e[1], std::log(4.0), 1e-8);
  
  param_type params({2, 3}, 5.0);
  ct_type f({x, y}, params);
  BOOST_CHECK(table_properties(f, {x, y}));
  BOOST_CHECK_EQUAL(boost::count(f, 5.0), 6);

  ct_type g({x}, {6.0, 6.5});
  BOOST_CHECK(table_properties(g, {x}));
  BOOST_CHECK_EQUAL(g[0], 6.0);
  BOOST_CHECK_EQUAL(g[1], 6.5);
}

BOOST_AUTO_TEST_CASE(test_assignment_swap) {
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 3);

  ct_type f;
  f = logd(2.0);
  BOOST_CHECK(table_properties(f, {}));
  BOOST_CHECK_CLOSE(f[0], std::log(2.0), 1e-8);
  
  f.reset({x, y});
  BOOST_CHECK(table_properties(f, {x, y}));

  f = logd(3.0);
  BOOST_CHECK(table_properties(f, {}));
  BOOST_CHECK_CLOSE(f[0], std::log(3.0), 1e-8);
  
  table_factor tf(make_vector(x), 0.5);
  f = tf;
  BOOST_CHECK(table_properties(f, {x}));
  BOOST_CHECK_CLOSE(f[0], std::log(0.5), 1e-8);
  BOOST_CHECK_CLOSE(f[1], std::log(0.5), 1e-8);

  ct_type g({x, y});
  swap(f, g);
  BOOST_CHECK(table_properties(f, {x, y}));
  BOOST_CHECK(table_properties(g, {x}));
}


BOOST_AUTO_TEST_CASE(test_indexing) {
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 3);
  
  ct_type f({x, y});
  std::iota(f.begin(), f.end(), 1);
  BOOST_CHECK_CLOSE(f(finite_index{0,0}).lv, 1.0, 1e-8);
  BOOST_CHECK_CLOSE(f(finite_index{1,0}).lv, 2.0, 1e-8);
  BOOST_CHECK_CLOSE(f(finite_index{0,1}).lv, 3.0, 1e-8);
  BOOST_CHECK_CLOSE(f(finite_index{1,1}).lv, 4.0, 1e-8);
  BOOST_CHECK_CLOSE(f(finite_index{0,2}).lv, 5.0, 1e-8);
  BOOST_CHECK_CLOSE(f(finite_index{1,2}).lv, 6.0, 1e-8);

  BOOST_CHECK_CLOSE(f(finite_assignment{{x,0}, {y,0}}).lv, 1.0, 1e-8);
  BOOST_CHECK_CLOSE(f(finite_assignment{{x,1}, {y,0}}).lv, 2.0, 1e-8);
  BOOST_CHECK_CLOSE(f(finite_assignment{{x,0}, {y,1}}).lv, 3.0, 1e-8);
  BOOST_CHECK_CLOSE(f(finite_assignment{{x,1}, {y,1}}).lv, 4.0, 1e-8);
  BOOST_CHECK_CLOSE(f(finite_assignment{{x,0}, {y,2}}).lv, 5.0, 1e-8);
  BOOST_CHECK_CLOSE(f(finite_assignment{{x,1}, {y,2}}).lv, 6.0, 1e-8);

  BOOST_CHECK_CLOSE(f.log(finite_index{0,2}), 5.0, 1e-8);
  BOOST_CHECK_CLOSE(f.log(finite_assignment{{x,0},{y,2}}), 5.0, 1e-8);

  finite_assignment a;
  f.assignment({1, 2}, a);
  BOOST_CHECK_EQUAL(a[x], 1);
  BOOST_CHECK_EQUAL(a[y], 2);
  BOOST_CHECK_EQUAL(f.index(a), 5);

  finite_variable* v = u.new_finite_variable("v", 2);
  finite_variable* w = u.new_finite_variable("w", 3);
  f.subst_args({{x, v}, {y, w}});
  BOOST_CHECK(table_properties(f, {v, w}));
}


BOOST_AUTO_TEST_CASE(test_operators) {
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 2);
  finite_variable* z = u.new_finite_variable("z", 3);

  ct_type f({x, y}, {0, 1, 2, 3});
  ct_type g({y, z}, {1, 2, 3, 4, 5, 6});
  ct_type h;
  h = f * g;
  BOOST_CHECK(table_properties(h, {x, y, z}));
  for (const finite_assignment& a : assignments({x, y, z})) {
    BOOST_CHECK_CLOSE(h.log(a), f.log(a) + g.log(a), 1e-8);
  }

  h *= g;
  BOOST_CHECK(table_properties(h, {x, y, z}));
  for (const finite_assignment& a : assignments({x, y, z})) {
    BOOST_CHECK_CLOSE(h.log(a), f.log(a) + 2*g.log(a), 1e-8);
  }

  h = f / g;
  BOOST_CHECK(table_properties(h, {x, y, z}));
  for (const finite_assignment& a : assignments({x, y, z})) {
    BOOST_CHECK_CLOSE(h.log(a), f.log(a) - g.log(a), 1e-8);
  }

  h /= f;
  BOOST_CHECK(table_properties(h, {x, y, z}));
  for (const finite_assignment& a : assignments({x, y, z})) {
    BOOST_CHECK_CLOSE(h.log(a), -g.log(a), 1e-8);
  }

  h = f * logd(2.0, log_tag());
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const finite_assignment& a : assignments({x, y})) {
    BOOST_CHECK_CLOSE(h.log(a), f.log(a) + 2.0, 1e-8);
  }

  h *= logd(1.0, log_tag());
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const finite_assignment& a : assignments({x, y})) {
    BOOST_CHECK_CLOSE(h.log(a), f.log(a) + 3.0, 1e-8);
  }

  h = logd(2.0, log_tag()) * f;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const finite_assignment& a : assignments({x, y})) {
    BOOST_CHECK_CLOSE(h.log(a), f.log(a) + 2.0, 1e-8);
  }

  h /= logd(1.0, log_tag());
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const finite_assignment& a : assignments({x, y})) {
    BOOST_CHECK_CLOSE(h.log(a), f.log(a) + 1.0, 1e-8);
  }

  h = f / logd(2.0, log_tag());
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const finite_assignment& a : assignments({x, y})) {
    BOOST_CHECK_CLOSE(h.log(a), f.log(a) - 2.0, 1e-8);
  }
  
  h = logd(2.0, log_tag()) / f;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const finite_assignment& a : assignments({x, y})) {
    BOOST_CHECK_CLOSE(h.log(a), 2.0 - f.log(a), 1e-8);
  }

  h = pow(f, 2.0);
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const finite_assignment& a : assignments({x, y})) {
    BOOST_CHECK_CLOSE(h.log(a), 2.0 * f.log(a), 1e-8);
  }
  
  ct_type f1({x, y}, {0, 1, 2, 3});
  ct_type f2({x, y}, {-2, 3, 0, 0});
  std::vector<double> fmax = {0, 3, 2, 3};
  std::vector<double> fmin = {-2, 1, 0, 0};

  h = max(f1, f2);
  BOOST_CHECK(table_properties(h, {x, y}));
  BOOST_CHECK(boost::equal(h, fmax));

  h = min(f1, f2);
  BOOST_CHECK(table_properties(h, {x, y}));
  BOOST_CHECK(boost::equal(h, fmin));

  h = weighted_update(f1, f2, 0.3);
  for (size_t i = 0; i < 4; ++i) {
    BOOST_CHECK_CLOSE(h[i], 0.7 * f1[i] + 0.3 * f2[i], 1e-8);
  }  
}


BOOST_AUTO_TEST_CASE(test_collapse) {  
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 3);

  ct_type f({x, y}, {0, 1, 2, 3, 5, 6});
  ct_type h;
  finite_assignment a;

  std::vector<double> hmax = {1, 3, 6};
  std::vector<double> hmin = {0, 2, 5};

  h = f.maximum({y});
  BOOST_CHECK(table_properties(h, {y}));
  BOOST_CHECK(boost::equal(h, hmax));
  BOOST_CHECK_EQUAL(f.maximum().lv, 6.0);
  BOOST_CHECK_EQUAL(f.maximum(a).lv, 6.0);
  BOOST_CHECK_EQUAL(a[x], 1);
  BOOST_CHECK_EQUAL(a[y], 2);
  
  h = f.minimum({y});
  BOOST_CHECK(table_properties(h, {y}));
  BOOST_CHECK(boost::equal(h, hmin));
  BOOST_CHECK_EQUAL(f.minimum().lv, 0.0);
  BOOST_CHECK_EQUAL(f.minimum(a).lv, 0.0);
  BOOST_CHECK_EQUAL(a[x], 0);
  BOOST_CHECK_EQUAL(a[y], 0);

  double pxy[] = {1.1, 0.5, 0.1, 0.2, 0.4, 0.0};
  double py[] = {1.6, 0.3, 0.4};
  ct_type g({x, y});
  std::transform(pxy, pxy + 6, g.begin(), logarithm<double>());
  h = g.marginal({y});
  BOOST_CHECK(table_properties(h, {y}));
  for (size_t i = 0; i < 3; ++i) {
    BOOST_CHECK_CLOSE(std::exp(h[i]), py[i], 1e-7);
  }
  BOOST_CHECK_CLOSE(double(g.marginal()), std::accumulate(pxy, pxy + 6, 0.0), 1e-8);
  BOOST_CHECK_CLOSE(double(h.normalize().marginal()), 1.0, 1e-8);
}


BOOST_AUTO_TEST_CASE(test_restrict) {
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 3);

  ct_type f({x, y}, {0, 1, 2, 3, 5, 6});
  ct_type h = f.restrict({{x, 1}});
  std::vector<double> fr = {1, 3, 6};
  BOOST_CHECK(table_properties(h, {y}));
  BOOST_CHECK(boost::equal(h, fr));
}


BOOST_AUTO_TEST_CASE(test_entropy) {
  using std::log;
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 2);

  std::vector<double> pxy = {0.1, 0.2, 0.3, 0.4};
  std::vector<double> qxy = {0.4*0.3, 0.6*0.3, 0.4*0.7, 0.6*0.7};
  ct_type p(make_dense_table_factor({x, y}, pxy));
  ct_type q(make_dense_table_factor({x, y}, qxy));
  ct_type m = (p+q) / logd(2);
  double hpxy = -(0.1*log(0.1) + 0.2*log(0.2) + 0.3*log(0.3) + 0.4*log(0.4));
  double hpx = -(0.4*log(0.4) + 0.6*log(0.6));
  double hpy = -(0.3*log(0.3) + 0.7*log(0.7));
  double hpq = 0.0, klpq = 0.0, sumdiff = 0.0, maxdiff = 0.0;
  for (size_t i = 0; i < 4; ++i) {
    hpq += -pxy[i] * log(qxy[i]);
    klpq += pxy[i] * log(pxy[i]/qxy[i]);
    double diff = std::abs(std::log(pxy[i]) - std::log(qxy[i]));
    sumdiff += diff;
    maxdiff = std::max(maxdiff, diff);
  }
  double jspq = (kl_divergence(p, m) + kl_divergence(q, m)) / 2;
  BOOST_CHECK_CLOSE(p.entropy(), hpxy, 1e-6);
  BOOST_CHECK_CLOSE(p.entropy({x}), hpx, 1e-6);
  BOOST_CHECK_CLOSE(p.entropy({y}), hpy, 1e-6);
  BOOST_CHECK_CLOSE(p.mutual_information({x}, {y}), klpq, 1e-6);
  BOOST_CHECK_CLOSE(cross_entropy(p, q), hpq, 1e-6);
  BOOST_CHECK_CLOSE(kl_divergence(p, q), klpq, 1e-6);
  BOOST_CHECK_CLOSE(js_divergence(p, q), jspq, 1e-6);
  BOOST_CHECK_CLOSE(sum_diff(p, q), sumdiff, 1e-6);
  BOOST_CHECK_CLOSE(max_diff(p, q), maxdiff, 1e-6);
}


bool is_close(const param_type& p, double v0, double v1) {
  return std::abs(p[0] - v0) < 1e-8 && std::abs(p[1] - v1) < 1e-8;
}

BOOST_AUTO_TEST_CASE(test_param) {
  const param_type p({1, 2}, {1, 2});
  const param_type q({1, 2}, {1.5, -0.5});
  param_type r;
  
  r = p; r += q;
  BOOST_CHECK(is_close(r, 2.5, 1.5));
  
  r = p; r -= q;
  BOOST_CHECK(is_close(r, -0.5, 2.5));

  r = p; r += 1;
  BOOST_CHECK(is_close(r, 2, 3));

  r = p; r -= 1;
  BOOST_CHECK(is_close(r, 0, 1));

  r = p; r *= 2;
  BOOST_CHECK(is_close(r, 2, 4));
  
  r = p; r /= 2;
  BOOST_CHECK(is_close(r, 0.5, 1));

  r = p; axpy(2, q, r);
  BOOST_CHECK(is_close(r, 4, 1));
  
  BOOST_CHECK_CLOSE(p.max(), 2, 1e-8);
  BOOST_CHECK_CLOSE(p.min(), 1, 1e-8);
  BOOST_CHECK_CLOSE(dot(p, q), 0.5, 1e-8);

  param_type cond = p.condition({1});
  BOOST_CHECK_EQUAL(cond.arity(), 1);
  BOOST_CHECK_EQUAL(cond.size(), 1);
  BOOST_CHECK_EQUAL(cond[0], 2);
}

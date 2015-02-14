#define BOOST_TEST_MODULE probability_array
#include <boost/test/unit_test.hpp>

#include <sill/base/finite_assignment_iterator.hpp>
#include <sill/base/universe.hpp>
#include <sill/factor/canonical_array.hpp>
#include <sill/factor/probability_array.hpp>
#include <sill/factor/probability_table.hpp>

#include <boost/range/algorithm.hpp>

#include "predicates.hpp"

namespace sill {
  template class probability_array<double>;
  template class probability_array<float>;
}

using namespace sill;

typedef probability_array<double> pa_type;
typedef pa_type::param_type param_type;
typedef logarithmic<double> logd;

BOOST_AUTO_TEST_CASE(test_constructors) {
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 3);

  pa_type a;
  BOOST_CHECK(a.empty());
  BOOST_CHECK(a.arguments().empty());

  pa_type b({x, y});
  BOOST_CHECK(table_properties(b, {x, y}));

  pa_type c(2.0);
  BOOST_CHECK(table_properties(c, {}));
  BOOST_CHECK_CLOSE(c[0], 2.0, 1e-8);
  
  pa_type d({x}, 3.0);
  BOOST_CHECK(table_properties(d, {x}));
  BOOST_CHECK_CLOSE(d[0], 3.0, 1e-8);
  BOOST_CHECK_CLOSE(d[1], 3.0, 1e-8);

  param_type params(2, 3);
  params.fill(5.0);
  pa_type f({x, y}, params);
  BOOST_CHECK(table_properties(f, {x, y}));
  BOOST_CHECK_EQUAL(boost::count(f, 5.0), 6);

  pa_type g({x}, {6.0, 6.5});
  BOOST_CHECK(table_properties(g, {x}));
  BOOST_CHECK_EQUAL(g[0], 6.0);
  BOOST_CHECK_EQUAL(g[1], 6.5);
}

BOOST_AUTO_TEST_CASE(test_assignment_swap) {
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 3);

  pa_type f;
  f = 2.0;
  BOOST_CHECK(table_properties(f, {}));
  BOOST_CHECK_CLOSE(f[0], 2.0, 1e-8);
  
  f.reset({x, y});
  BOOST_CHECK(table_properties(f, {x, y}));

  f = 3.0;
  BOOST_CHECK(table_properties(f, {}));
  BOOST_CHECK_CLOSE(f[0], 3.0, 1e-8);
  
  canonical_array<> ca({x}, {0.5, 0.7});
  f = ca;
  BOOST_CHECK(table_properties(f, {x}));
  BOOST_CHECK_CLOSE(f[0], std::exp(0.5), 1e-8);
  BOOST_CHECK_CLOSE(f[1], std::exp(0.7), 1e-8);

  probability_table<double> pt(make_vector(x), 0.5);
  f = pt;
  BOOST_CHECK(table_properties(f, {x}));
  BOOST_CHECK_CLOSE(f[0], 0.5, 1e-8);
  BOOST_CHECK_CLOSE(f[1], 0.5, 1e-8);

  pa_type g({x, y});
  swap(f, g);
  BOOST_CHECK(table_properties(f, {x, y}));
  BOOST_CHECK(table_properties(g, {x}));
}


BOOST_AUTO_TEST_CASE(test_indexing) {
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 3);
  
  pa_type f({x, y});
  std::iota(f.begin(), f.end(), 1);
  BOOST_CHECK_CLOSE(f(finite_index{0,0}), 1.0, 1e-8);
  BOOST_CHECK_CLOSE(f(finite_index{1,0}), 2.0, 1e-8);
  BOOST_CHECK_CLOSE(f(finite_index{0,1}), 3.0, 1e-8);
  BOOST_CHECK_CLOSE(f(finite_index{1,1}), 4.0, 1e-8);
  BOOST_CHECK_CLOSE(f(finite_index{0,2}), 5.0, 1e-8);
  BOOST_CHECK_CLOSE(f(finite_index{1,2}), 6.0, 1e-8);

  BOOST_CHECK_CLOSE(f(finite_assignment{{x,0}, {y,0}}), 1.0, 1e-8);
  BOOST_CHECK_CLOSE(f(finite_assignment{{x,1}, {y,0}}), 2.0, 1e-8);
  BOOST_CHECK_CLOSE(f(finite_assignment{{x,0}, {y,1}}), 3.0, 1e-8);
  BOOST_CHECK_CLOSE(f(finite_assignment{{x,1}, {y,1}}), 4.0, 1e-8);
  BOOST_CHECK_CLOSE(f(finite_assignment{{x,0}, {y,2}}), 5.0, 1e-8);
  BOOST_CHECK_CLOSE(f(finite_assignment{{x,1}, {y,2}}), 6.0, 1e-8);

  BOOST_CHECK_CLOSE(f.log(finite_index{0,2}), std::log(5.0), 1e-8);
  BOOST_CHECK_CLOSE(f.log(finite_assignment{{x,0},{y,2}}), std::log(5.0), 1e-8);

  finite_assignment a;
  f.assignment(5, a);
  BOOST_CHECK_EQUAL(a[x], 1);
  BOOST_CHECK_EQUAL(a[y], 2);
  BOOST_CHECK_EQUAL(f.linear_index(a), 5);

  finite_variable* v = u.new_finite_variable("v", 2);
  finite_variable* w = u.new_finite_variable("w", 3);
  f.subst_args({{x, v}, {y, w}});
  BOOST_CHECK(table_properties(f, {v, w}));
}


BOOST_AUTO_TEST_CASE(test_operators) {
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 2);

  pa_type f({x, y}, {0, 1, 2, 3});
  pa_type g({y}, {3, 4});
  pa_type h;
  h = f * g;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const finite_assignment& a : assignments({x, y})) {
    BOOST_CHECK_CLOSE(h(a), f(a) * g(a), 1e-8);
  }

  h *= g;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const finite_assignment& a : assignments({x, y})) {
    BOOST_CHECK_CLOSE(h(a), f(a) * g(a) * g(a), 1e-8);
  }

  h = f / g;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const finite_assignment& a : assignments({x, y})) {
    BOOST_CHECK_CLOSE(h(a), f(a) / g(a), 1e-8);
  }

  h /= f;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const finite_assignment& a : assignments({x, y})) {
    BOOST_CHECK_CLOSE(h(a), f(a) ? (1.0 / g(a)) : 0.0, 1e-8);
  }

  h = f * 2.0;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const finite_assignment& a : assignments({x, y})) {
    BOOST_CHECK_CLOSE(h(a), f(a) * 2.0, 1e-8);
  }

  h *= 3.0;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const finite_assignment& a : assignments({x, y})) {
    BOOST_CHECK_CLOSE(h(a), f(a) * 6.0, 1e-8);
  }

  h = 2.0 * f;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const finite_assignment& a : assignments({x, y})) {
    BOOST_CHECK_CLOSE(h(a), f(a) * 2.0, 1e-8);
  }

  h /= 4.0;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const finite_assignment& a : assignments({x, y})) {
    BOOST_CHECK_CLOSE(h(a), f(a) * 0.5, 1e-8);
  }

  h = f / 3.0;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const finite_assignment& a : assignments({x, y})) {
    BOOST_CHECK_CLOSE(h(a), f(a) / 3.0, 1e-8);
  }
  
  h = 3.0 / f;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const finite_assignment& a : assignments({x, y})) {
    if (f(a)) {
      BOOST_CHECK_CLOSE(h(a), 3.0 / f(a), 1e-8);
    } else {
      BOOST_CHECK(std::isinf(h(a)));
    }
  }

  h = pow(f, 3.0);
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const finite_assignment& a : assignments({x, y})) {
    BOOST_CHECK_CLOSE(h(a), std::pow(f(a), 3.0), 1e-8);
  }
  
  pa_type f1({x, y}, {0, 1, 2, 3});
  pa_type f2({x, y}, {-2, 3, 0, 0});
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

  pa_type f({x, y}, {0, 1, 2, 3, 5, 6});
  pa_type h;
  finite_assignment a;

  std::vector<double> hmax = {1, 3, 6};
  std::vector<double> hmin = {0, 2, 5};

  h = f.maximum({y});
  BOOST_CHECK(table_properties(h, {y}));
  BOOST_CHECK(boost::equal(h, hmax));
  BOOST_CHECK_EQUAL(f.maximum(), 6.0);
  BOOST_CHECK_EQUAL(f.maximum(a), 6.0);
  BOOST_CHECK_EQUAL(a[x], 1);
  BOOST_CHECK_EQUAL(a[y], 2);
  
  h = f.minimum({y});
  BOOST_CHECK(table_properties(h, {y}));
  BOOST_CHECK(boost::equal(h, hmin));
  BOOST_CHECK_EQUAL(f.minimum(), 0.0);
  BOOST_CHECK_EQUAL(f.minimum(a), 0.0);
  BOOST_CHECK_EQUAL(a[x], 0);
  BOOST_CHECK_EQUAL(a[y], 0);

  pa_type pxy({x, y}, {1.1, 0.5, 0.1, 0.2, 0.4, 0.0});
  pa_type py({y}, {1.6, 0.3, 0.4});
  h = pxy.marginal({y});
  BOOST_CHECK(table_properties(h, {y}));
  for (size_t i = 0; i < 3; ++i) {
    BOOST_CHECK_CLOSE(h[i], py[i], 1e-7);
  }
  BOOST_CHECK_CLOSE(pxy.marginal(), 1.1+0.5+0.1+0.2+0.4, 1e-8);
  BOOST_CHECK_CLOSE(h.normalize().marginal(), 1.0, 1e-8);
}


BOOST_AUTO_TEST_CASE(test_restrict) {
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 3);

  pa_type f({x, y}, {0, 1, 2, 3, 5, 6});
  pa_type h = f.restrict({{x, 1}});
  std::vector<double> fr = {1, 3, 6};
  BOOST_CHECK(table_properties(h, {y}));
  BOOST_CHECK(boost::equal(h, fr));
}


BOOST_AUTO_TEST_CASE(test_entropy) {
  using std::log;
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 2);

  pa_type p({x, y}, {0.1, 0.2, 0.3, 0.4});
  pa_type q({x, y}, {0.4*0.3, 0.6*0.3, 0.4*0.7, 0.6*0.7});
  pa_type m = (p+q) / 2.0;
  double hpxy = -(0.1*log(0.1) + 0.2*log(0.2) + 0.3*log(0.3) + 0.4*log(0.4));
  double hpx = -(0.4*log(0.4) + 0.6*log(0.6));
  double hpy = -(0.3*log(0.3) + 0.7*log(0.7));
  double hpq = 0.0, klpq = 0.0, sumdiff = 0.0, maxdiff = 0.0;
  for (size_t i = 0; i < 4; ++i) {
    hpq += -p[i] * log(q[i]);
    klpq += p[i] * log(p[i]/q[i]);
    double diff = std::abs(p[i] - q[i]);
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

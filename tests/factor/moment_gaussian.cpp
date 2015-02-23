#define BOOST_TEST_MODULE moment_gaussian
#include <boost/test/unit_test.hpp>

#include <sill/factor/moment_gaussian.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/canonical_gaussian.hpp>

#include "predicates.hpp"
#include "../math/eigen/helpers.hpp"

namespace sill {
  template class moment_gaussian<double>;
  template class moment_gaussian<float>;
  template class moment_gaussian_param<double>;
  template class moment_gaussian_param<float>;
}

using namespace sill;

typedef moment_gaussian_param<double> param_type;
typedef moment_gaussian<double> mg_type;
typedef canonical_gaussian<double> cg_type;
typedef dynamic_vector<double> vec_type;
typedef dynamic_matrix<double> mat_type;
typedef logarithmic<double> logd;

boost::test_tools::predicate_result
mg_properties(const mg_type& f,
              const domain<vector_variable*>& head,
              const domain<vector_variable*>& tail = domain<vector_variable*>()) {
  size_t m = vector_size(head);
  size_t n = vector_size(tail);
  domain<vector_variable*> args = head + tail;

  if (f.empty() && !args.empty()) {
    boost::test_tools::predicate_result result(false);
    result.message() << "The factor is empty [" << f << "]";
    return result;
  }

  if (f.arity() != args.size()) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid factor arity ["
                     << f.arity() << " != " << args.size() << "]";
    return result;
  }
  if (f.head_arity() != head.size()) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid head arity ["
                     << f.head_arity() << " != " << head.size() << "]";
    return result;
  }
  if (f.tail_arity() != tail.size()) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid factor arity ["
                     << f.tail_arity() << " != " << head.size() << "]";
    return result;
  }

  if (f.size() != m + n) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid factor size ["
                     << f.size() << " != " << (m + n) << "]";
    return result;
  }
  if (f.head_size() != m) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid head size ["
                     << f.head_size() << " != " << m << "]";
    return result;
  }
  if (f.tail_size() != n) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid tail size ["
                     << f.size() << " != " << n << "]";
    return result;
  }

  if (f.arguments() != args) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid factor domain ["
                     << f.arguments() << " != " << args << "]";
    return result;
  }
  if (f.head() != head) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid factor head ["
                     << f.head() << " != " << head << "]";
    return result;
  }
  if (f.tail() != tail) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid factor tail ["
                     << f.tail() << " != " << tail << "]";
    return result;
  }

  return true;
}

boost::test_tools::predicate_result
mg_params(const mg_type& f,
          const vec_type& mean,
          const mat_type& cov,
          const mat_type& coef,
          double lm) {
  if (!f.mean().isApprox(mean, 1e-8)) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Means not close: "
                     << f.mean().transpose() << " vs "
                     << mean.transpose();
    return result;
  }
  if (!f.covariance().isApprox(cov, 1e-8)) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Covariance matrices not close: \n"
                     << f.covariance() << "\n vs \n" << cov;
    return result;
  }
  if (!f.coefficients().isApprox(coef, 1e-8)) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Coefficient matrices not close: \n"
                     << f.coefficients() << "\n vs \n" << coef;
    return result;
  }
  if (std::abs(f.log_multiplier() - lm) > 1e-8) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Log-multipliers not close: "
                     << f.log_multiplier() << " vs " << lm;
    return result;
  }
  return true;
}

BOOST_AUTO_TEST_CASE(test_constructors) {
  universe u;
  vector_variable* x = u.new_vector_variable("x", 2);
  vector_variable* y = u.new_vector_variable("y", 1);
  vector_variable* z = u.new_vector_variable("z", 1);

  mg_type a;
  BOOST_CHECK(a.empty());
  BOOST_CHECK(mg_properties(a, {}, {}));

  mg_type b({x, y});
  BOOST_CHECK(mg_properties(b, {x, y}));

  mg_type c({x, y}, {z});
  BOOST_CHECK(mg_properties(c, {x, y}, {z}));

  mg_type d(logd(2.0));
  BOOST_CHECK(mg_properties(d, {}, {}));
  BOOST_CHECK_CLOSE(d.log_multiplier(), std::log(2.0), 1e-8);
  
  param_type param(3, 0);
  param.mean = vec3(1, 2, 3);
  param.cov  = mat_type::Identity(3, 3);
  param.lm   = 2.0;
  mg_type e({x, y}, param);
  BOOST_CHECK(mg_properties(e, {x, y}));
  BOOST_CHECK(mg_params(e, vec3(1,2,3), mat_type::Identity(3, 3), mat_type(3, 0), 2.0));
  mg_type em({x, y}, std::move(param));
  BOOST_CHECK(mg_properties(em, {x, y}));
  BOOST_CHECK(mg_params(em, vec3(1,2,3), mat_type::Identity(3, 3), mat_type(3, 0), 2.0));

  param = e.param();
  param.coef = vec3(0.5, 1, 2);
  mg_type f({x, y}, {z}, param);
  BOOST_CHECK(mg_properties(f, {x, y}, {z}));
  BOOST_CHECK(mg_params(f, vec3(1,2,3), mat_type::Identity(3, 3), vec3(0.5,1,2), 2.0));
  mg_type fm({x, y}, {z}, std::move(param));
  BOOST_CHECK(mg_properties(fm, {x, y}, {z}));
  BOOST_CHECK(mg_params(fm, vec3(1,2,3), mat_type::Identity(3, 3), vec3(0.5,1,2), 2.0));

  mg_type g({x, y}, vec3(3,2,1), mat33(2,1,1,1,2,1,1,1,3), 1.5);
  BOOST_CHECK(mg_properties(g, {x, y}));
  BOOST_CHECK(mg_params(g, vec3(3,2,1), mat33(2,1,1,1,2,1,1,1,3), mat_type(3, 0), 1.5));

  mg_type h({x, y}, {z}, vec3(3,2,1), mat33(2,1,1,1,2,1,1,1,3), vec3(4,5,6), 1.5);
  BOOST_CHECK(mg_properties(h, {x, y}, {z}));
  BOOST_CHECK(mg_params(h, vec3(3,2,1), mat33(2,1,1,1,2,1,1,1,3), vec3(4,5,6), 1.5));
}

BOOST_AUTO_TEST_CASE(test_assignment_swap) {
  universe u;
  vector_variable* x = u.new_vector_variable("x", 2);
  vector_variable* y = u.new_vector_variable("y", 1);
  vector_variable* z = u.new_vector_variable("z", 1);

  mg_type f;
  f = logd(2.0);
  BOOST_CHECK(mg_properties(f, {}));
  BOOST_CHECK_CLOSE(f.log_multiplier(), std::log(2.0), 1e-8);
  
  f.reset({x, y});
  BOOST_CHECK(mg_properties(f, {x, y}));

  f = logd(3.0);
  BOOST_CHECK(mg_properties(f, {}));
  BOOST_CHECK_CLOSE(f.log_multiplier(), std::log(3.0), 1e-8);

  cg_type cg({x}, vec2(1,2), mat22(2,0,0,2), 1.5);
  f = cg;
  double lm = 1.5 + std::log(two_pi<double>()) - 0.5*std::log(4) + 0.5*2.5;
  BOOST_CHECK(mg_properties(f, {x}));
  BOOST_CHECK(mg_params(f, vec2(0.5,1), mat22(0.5,0,0,0.5), mat_type(2,0), lm));
  BOOST_CHECK_CLOSE(f.maximum().lv, cg.maximum().lv, 1e-8);
  BOOST_CHECK_CLOSE(f.marginal().lv, cg.marginal().lv, 1e-8);
  BOOST_CHECK_CLOSE(f(vec2(1,-3)).lv, cg(vec2(1,-3)).lv, 1e-8);

  mg_type g({x, y});
  swap(f, g);
  BOOST_CHECK(mg_properties(f, {x, y}));
  BOOST_CHECK(mg_properties(g, {x}));
}


BOOST_AUTO_TEST_CASE(test_indexing) {
  universe u;
  vector_variable* x = u.new_vector_variable("x", 2);
  vector_variable* y = u.new_vector_variable("y", 1);
  
  mg_type f({x, y}, vec3(2, 1, 0), 2*mat_type::Identity(3, 3), 0.5);
  vec_type vec = vec3(0.5, -2, 0);
  double lv = -0.25*(1.5*1.5+3*3)-1.5*log(two_pi<double>())-0.5*log(8)+0.5;
  BOOST_CHECK_CLOSE(f.log(vec), lv, 1e-8);

  vector_assignment<double> a;
  f.assignment(vec3(3, 2, 1), a);
  BOOST_CHECK_EQUAL(a[x], vec2(3, 2));
  BOOST_CHECK_EQUAL(a[y], vec1(1));

  vector_variable* v = u.new_vector_variable("v", 2);
  vector_variable* w = u.new_vector_variable("w", 1);
  f.subst_args({{x, v}, {y, w}});
  BOOST_CHECK(mg_properties(f, {v, w}));
}


BOOST_AUTO_TEST_CASE(test_multiplication) {
  universe u;
  vector_variable* x = u.new_vector_variable("x", 1);
  vector_variable* y = u.new_vector_variable("y", 2);
  vector_variable* z = u.new_vector_variable("z", 1);

  // small test
  mg_type f({x}, {y}, vec1(1), mat11(2), mat12(0.5, 3), 1.2);
  mg_type g({y}, vec2(2, 1), mat22(2,1,1,2), 0.3);

  mg_type h = f * g;
  BOOST_CHECK(mg_properties(h, {x, y}));
  BOOST_CHECK(mg_params(h, 
                        vec3(1+1+3,2,1),
                        mat33(2+0.5+3+18,1+3,0.5+6,1+3,2,1,0.5+6,1,2),
                        mat_type(3,0),
                        1.5));

  mg_type h2 = g * f;
  param_type p2 = h.reorder({y, x}).param();
  BOOST_CHECK(mg_properties(h2, {y, x}));
  BOOST_CHECK(mg_params(h2, p2.mean, p2.cov, p2.coef, p2.lm));

  // multiplication by a constant
  mg_type h3 = h2 * logd(2.0, log_tag());
  BOOST_CHECK(mg_properties(h3, {y, x}));
  BOOST_CHECK(mg_params(h3, p2.mean, p2.cov, p2.coef, p2.lm+2.0));

  mg_type h4 = logd(2.0, log_tag()) * h2;
  BOOST_CHECK(mg_properties(h4, {y, x}));
  BOOST_CHECK(mg_params(h4, p2.mean, p2.cov, p2.coef, p2.lm+2.0));
  h4 *= logd(3.0, log_tag());
  BOOST_CHECK_CLOSE(h4.log_multiplier(), p2.lm+5, 1e-8);

  // division by a constant
  mg_type h5 = h2 / logd(2.0, log_tag());
  BOOST_CHECK(mg_properties(h5, {y, x}));
  BOOST_CHECK(mg_params(h5, p2.mean, p2.cov, p2.coef, p2.lm-2.0));
  h5 /= logd(3.0, log_tag());
  BOOST_CHECK_CLOSE(h5.log_multiplier(), p2.lm-5, 1e-8);

  // large test (1, 2, 1)
  f = mg_type({x, y}, vec3(3,2,1), mat33(4,2,2,2,3,2,2,2,2.5), 1.5);
  g = mg_type({z}, {y}, vec1(0.5), mat11(0.8), mat12(0.1,0.2), 1.7);
  h = f * g;
  param_type p = (f.canonical() * g.canonical()).moment().param();
  BOOST_CHECK(mg_properties(h, {x, y, z}));
  BOOST_CHECK(mg_params(h, p.mean, p.cov, mat_type(4,0), p.lm));
}

BOOST_AUTO_TEST_CASE(test_collapse) {  
  universe u;
  vector_variable* x = u.new_vector_variable("x", 1);
  vector_variable* y = u.new_vector_variable("y", 1);
  vector_variable* z = u.new_vector_variable("z", 1);

  vec_type mean = vec3(2, 0.5, 0.2);
  mat_type cov = mat33(2, 1, 1, 1, 3, 1, 1, 1, 4);
  mg_type f({x, y, z}, mean, cov, 2.0);

  // test all marginal
  mg_type h = f;
  BOOST_CHECK(h.normalizable());
  h.normalize();
  BOOST_CHECK_SMALL(h.marginal().lv, 1e-8);

  // test block marginal
  h = f.marginal({x, y});
  matrix_index ind(0, 2);
  vec_type meanxy = mean.segment(0, 2);
  mat_type covxy = cov.block(0, 0, 2, 2);
  BOOST_CHECK(mg_properties(h, {x, y}));
  BOOST_CHECK(mg_params(h, meanxy, covxy, mat_type(2,0), f.log_multiplier()));
  BOOST_CHECK_CLOSE(h.marginal().lv, 2.0, 1e-8);

  // test plain marginal
  h = f.marginal({z, x});
  ind = {2, 0};
  vec_type meanzx = subvec(mean, ind).plain();
  mat_type covzx = submat(cov, ind, ind).plain();
  BOOST_CHECK(mg_properties(h, {z, x}));
  BOOST_CHECK(mg_params(h, meanzx, covzx, mat_type(2,0), f.log_multiplier()));
  BOOST_CHECK_CLOSE(h.marginal().lv, 2.0, 1e-8);

  // test conditional marginal
  mg_type g({y, x}, {z}, vec2(3,2), mat22(2,1,1,2), mat21(4,5), 1.2);
  h = g.marginal({x, z});
  BOOST_CHECK(mg_properties(h, {x}, {z}));
  BOOST_CHECK(mg_params(h, vec1(2), mat11(2), mat11(5), 1.2));

  // test maximum assignment
  vector_assignment<double> a;
  logd max = f.maximum(a);
  BOOST_CHECK_CLOSE(a[x][0], mean[0], 1e-8);
  BOOST_CHECK_CLOSE(a[y][0], mean[1], 1e-8);
  BOOST_CHECK_CLOSE(a[z][0], mean[2], 1e-8);
  BOOST_CHECK_CLOSE(f.log(mean), max.lv, 1e-8);
}


BOOST_AUTO_TEST_CASE(test_restrict) {
  universe u;
  vector_variable* x = u.new_vector_variable("x", 1);
  vector_variable* y = u.new_vector_variable("y", 1);
  vector_variable* z = u.new_vector_variable("z", 1);
  vector_variable* w = u.new_vector_variable("w", 1);

  mg_type f({x, y}, {z, w}, vec2(3,4), mat22(2,1,1,2), mat22(4,5,2,-1), 2.0);

  // restrict all tail and some head
  vector_assignment<double> a = {{y, vec1(1)}, {z, vec1(2)}, {w, vec1(3)}};
  mg_type g = f.restrict(a);
  param_type p = f.canonical().restrict(a).moment().param();
  BOOST_CHECK(mg_properties(g, {x}, {}));
  BOOST_CHECK(mg_params(g, p.mean, p.cov, p.coef, p.lm));

  // restrict some tail and no head
  a = {{w, vec1(2)}};
  g = f.restrict(a);
  BOOST_CHECK(mg_properties(g, {x,y}, {z}));
  BOOST_CHECK(mg_params(g, vec2(3+5*2,4-1*2), mat22(2,1,1,2), mat21(4,2), 2.0));
}


BOOST_AUTO_TEST_CASE(test_entropy) {
  universe u;
  vector_variable* x = u.new_vector_variable("x", 1);
  vector_variable* y = u.new_vector_variable("y", 1);
  vector_variable* z = u.new_vector_variable("z", 1);

  vec_type mean = vec3(2, 0.5, 0.2);
  mat_type cov = mat33(2, 1, 1, 1, 3, 1, 1, 1, 4);
  mg_type p({x, y, z}, mean, cov, 2.0);

  double l2pi = std::log(two_pi<double>());
  double ent_xyz = std::log(cov.determinant()) + 3.0 * (l2pi + 1.0);
  BOOST_CHECK_CLOSE(p.entropy(), ent_xyz / 2.0, 1e-5);

  double ent_x = std::log(cov.block(0, 0, 1, 1).determinant()) + l2pi + 1.0;
  BOOST_CHECK_CLOSE(p.entropy({x}), ent_x / 2.0, 1e-5);
  
  mg_type q({x, y, z}, mean, cov + mat_type::Identity(3, 3));
  BOOST_CHECK_GE(kl_divergence(p, q), 0.0);
  BOOST_CHECK_SMALL(kl_divergence(p, p), 1e-6);
}

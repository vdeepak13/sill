#define BOOST_TEST_MODULE softmax
#include <boost/test/unit_test.hpp>

#include <sill/math/function/softmax.hpp>

namespace sill {
  template class softmax<double>;
  template class softmax<float>;
}

using namespace Eigen;
using namespace sill;

typedef softmax<double>::vec_type vec_type;
typedef softmax<double>::mat_type mat_type;
typedef sparse_index<double> sp_vec_type;

vec_type vec2(double v0, double v1) {
  return (vec_type(2) << v0, v1).finished();
}

mat_type mat2(double v00, double v01, double v10, double v11) {
  return (mat_type(2, 2) << v00, v01, v10, v11).finished();
}

BOOST_AUTO_TEST_CASE(test_construct) {
  softmax<double> a(3, 4);
  BOOST_CHECK_EQUAL(a.num_labels(), 3);
  BOOST_CHECK_EQUAL(a.num_features(), 4);
  
  softmax<double> b(3, 4, 1.5);
  BOOST_CHECK_EQUAL(b.num_labels(), 3);
  BOOST_CHECK_EQUAL(b.num_features(), 4);
  BOOST_CHECK((b.weight().array() == 1.5).all());
  BOOST_CHECK((b.bias().array() == 1.5).all());

  softmax<double> c(mat_type(2, 3), vec_type(2));
  BOOST_CHECK_EQUAL(c.num_labels(), 2);
  BOOST_CHECK_EQUAL(c.num_features(), 3);
}

BOOST_AUTO_TEST_CASE(test_value) {
  softmax<double> f(mat2(1, 2, 3, 4), vec2(-1, 1));
  vec_type pd = f(vec2(0.5, 0.7));
  vec_type ps = f({{1, 0.7}, {0, 0.5}});
  vec_type r = vec2(std::exp(0.5 * 1 + 0.7 * 2 - 1),
                    std::exp(0.5 * 3 + 0.7 * 4 + 1));
  r /= r.sum();
  BOOST_CHECK_CLOSE(pd[0], r[0], 1e-6);
  BOOST_CHECK_CLOSE(pd[1], r[1], 1e-6);
  BOOST_CHECK_CLOSE(ps[0], r[0], 1e-6);
  BOOST_CHECK_CLOSE(ps[1], r[1], 1e-6);
}

BOOST_AUTO_TEST_CASE(test_derivatives) {
  softmax<double> f(mat2(1, 2, 3, 4), vec2(-1, 1));
  vec_type p = vec2(0.2, 0.8);
  vec_type xd = vec2(0.5, 0.7);
  sp_vec_type xs = {{1, 0.7}, {0, 0.5}};
  double w = 1.5;
  double eps = 1e-4;

  // compute exact derivatives
  softmax<double> g1d(2, 2, 0.0);
  softmax<double> g1s(2, 2, 0.0);
  softmax<double> gpd(2, 2, 0.0);
  softmax<double> gps(2, 2, 0.0);
  softmax<double> hd(2, 2, 0.0);
  softmax<double> hs(2, 2, 0.0);
  f.add_gradient(1, xd, w, g1d);
  f.add_gradient(1, xs, w, g1s);
  f.add_gradient(p, xd, w, gpd);
  f.add_gradient(p, xs, w, gps);
  f.add_hessian_diag(xd, w, hd);
  f.add_hessian_diag(xs, w, hs);

  // compute approximate derivatives w.r.t. weights
  double f0 = f.log(xd)(1);
  double e0 = f.log(xd).dot(p);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      f.weight(i, j) += eps;
      double f1 = f.log(xd)(1);
      double e1 = f.log(xd).dot(p);
      f.weight(i, j) += eps;
      double f2 = f.log(xd)(1);
      f.weight(i, j) -= 2 * eps;
      double g1ij = (f1 - f0) / eps * w;
      double gpij = (e1 - e0) / eps * w;
      double hij = (f2 - 2*f1 + f0) / (eps * eps) * w;
      BOOST_CHECK_CLOSE(g1d.weight(i, j), g1ij, 1e-2);
      BOOST_CHECK_CLOSE(g1s.weight(i, j), g1ij, 1e-2);
      BOOST_CHECK_CLOSE(gpd.weight(i, j), gpij, 1e-2);
      BOOST_CHECK_CLOSE(gps.weight(i, j), gpij, 1e-2);
      BOOST_CHECK_CLOSE(hd.weight(i, j), hij, 1e-2);
      BOOST_CHECK_CLOSE(hs.weight(i, j), hij, 1e-2);
    }
  }

  // compute approximate derivatives w.r.t. biases
  for (size_t i = 0; i < 2; ++i) {
    f.bias(i) += eps;
    double f1 = f.log(xd)(1);
    double e1 = f.log(xd).dot(p);
    f.bias(i) += eps;
    double f2 = f.log(xd)(1);
    f.bias(i) -= 2 * eps;
    double g1ij = (f1 - f0) / eps * w;
    double gpij = (e1 - e0) / eps * w;
    double hij = (f2 - 2*f1 + f0) / (eps * eps) * w;
    BOOST_CHECK_CLOSE(g1d.bias(i), g1ij, 1e-2);
    BOOST_CHECK_CLOSE(g1s.bias(i), g1ij, 1e-2);
    BOOST_CHECK_CLOSE(gpd.bias(i), gpij, 1e-2);
    BOOST_CHECK_CLOSE(gps.bias(i), gpij, 1e-2);
    BOOST_CHECK_CLOSE(hd.bias(i), hij, 1e-2);
    BOOST_CHECK_CLOSE(hs.bias(i), hij, 1e-2);
  }
}

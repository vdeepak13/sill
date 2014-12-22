#define BOOST_TEST_MODULE slope_binary_search
#include <boost/test/unit_test.hpp>

#include <sill/optimization/line_search/slope_binary_search.hpp>

#include <boost/bind.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include "../quadratic_objective.hpp"

using namespace sill;
typedef line_search_result<double> result_type;

template class slope_binary_search<vec_type>;

// test the high-accuracy search
BOOST_AUTO_TEST_CASE(test_slope_binary_search) {
  quadratic_objective objective("5 4", "1 0; 0 1");
  slope_binary_search<vec_type> search;
  search.reset(boost::bind(&quadratic_objective::value, &objective, _1),
               boost::bind(&quadratic_objective::gradient, &objective, _1));
  result_type horiz = search.step("3.987 3", "1 0");
  BOOST_CHECK_CLOSE(horiz.step, 1.013, 1e-3);
  BOOST_CHECK_CLOSE(horiz.value, 0.5, 1e-3);
  result_type diag = search.step("1 0", "1 1");
  BOOST_CHECK_CLOSE(diag.step, 4.0, 1e-3);
  BOOST_CHECK_SMALL(diag.value, 1e-5);
}

// test that we reach Wolfe conditions
// by shooting from random points and verifying the conditions manually
BOOST_AUTO_TEST_CASE(test_wolfe) {
  quadratic_objective objective("-1 1", "2 1; 1 2");
  typedef wolfe_conditions<double>::param_type wolfe_param_type;
  wolfe_param_type wolfe = wolfe_param_type::conjugate_gradient();
  slope_binary_search<vec_type> search(wolfe);
  search.reset(boost::bind(&quadratic_objective::value, &objective, _1),
               boost::bind(&quadratic_objective::gradient, &objective, _1));
  
  size_t nlines = 20;
  boost::random::mt19937 rng;
  boost::random::uniform_real_distribution<> unif(-5, 5);
  for (size_t i = 0; i < nlines; ++i) {
    vec_type src(2), dir(2);
    src[0] = unif(rng);
    src[1] = unif(rng);
    dir[0] = unif(rng);
    dir[1] = unif(rng); // TODO: switch to list initializers with C++11
    if (dot(objective.gradient(src), dir) > 0) {
      dir = -dir;
    }
    double f0 = objective.value(src);
    double g0 = dot(objective.gradient(src), dir);
    result_type r = search.step(src, dir);
    double fa = objective.value(src+r.step*dir);
    double ga = dot(objective.gradient(src+r.step*dir), dir);

    // verify the accuracy of results and validity of the Wolfe conditions
    BOOST_CHECK_LT(r.value, objective.value(src));
    BOOST_CHECK_CLOSE(r.value, fa, 1e-2);
    BOOST_CHECK_LE(fa, f0 + wolfe.c1 * r.step * g0);
    BOOST_CHECK_LE(std::fabs(ga), wolfe.c2 * std::fabs(g0));
  }
}

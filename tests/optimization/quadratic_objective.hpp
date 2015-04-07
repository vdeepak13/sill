#ifndef SILL_TEST_QUADRATIC_OBJECTIVE_HPP
#define SILL_TEST_QUADRATIC_OBJECTIVE_HPP

#include <sill/optimization/gradient_objective.hpp>

#include <sill/math/eigen/dynamic.hpp>
#include <sill/math/eigen/optimization.hpp>

#include "../math/eigen/helpers.hpp"

typedef sill::dynamic_matrix<double> mat_type;
typedef sill::dynamic_vector<double> vec_type;

// a quadratic objective 0.5 * (x-ctr)^T cov (x-ctr)
struct quadratic_objective
  : public sill::gradient_objective<vec_type> {

  vec_type ctr;
  mat_type cov;
  vec_type g;
  vec_type h;

  quadratic_objective(const vec_type& ctr, const mat_type& cov)
    : ctr(ctr), cov(cov) { }
  
  double value(const vec_type& x) {
    vec_type diff = x - ctr;
    return 0.5 * diff.dot(cov * diff);
  }

  const vec_type& gradient(const vec_type& x) {
    g = cov * (x - ctr);
    return g;
  }

  const vec_type& hessian_diag(const vec_type& x) {
    h = cov.diagonal();
    return h;
  }
  
}; // struct quadratic_objective

#endif

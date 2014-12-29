#ifndef SILL_TEST_QUADRATIC_OBJECTIVE_HPP
#define SILL_TEST_QUADRATIC_OBJECTIVE_HPP

#include <sill/optimization/gradient_objective.hpp>

#include <armadillo>

typedef arma::mat mat_type;
typedef arma::vec vec_type;

// a quadratic objective 0.5 * (x-ctr)^T cov (x-ctr)
struct quadratic_objective
  : public sill::gradient_objective<vec_type> {

  vec_type ctr;
  mat_type cov;
  vec_type g;
  vec_type p;

  quadratic_objective(const vec_type& ctr, const mat_type& cov)
    : ctr(ctr), cov(cov) { }
  
  double value(const vec_type& x) {
    vec_type diff = x - ctr;
    return 0.5 * dot(diff, cov * diff);
  }

  const vec_type& gradient(const vec_type& x) {
    g = cov * (x - ctr);
    return g;
  }

  const vec_type& precondg(const vec_type& x) {
    p = (cov * (x - ctr)) / diagvec(cov);
    return p;
  }
  
}; // struct quadratic_objective


#endif

#ifndef SILL_TEST_QUADRATIC_OBJECTIVE_HPP
#define SILL_TEST_QUADRATIC_OBJECTIVE_HPP

#include <armadillo>

typedef arma::mat mat_type;
typedef arma::vec vec_type;

// a quadratic objective 0.5 * (x-ctr)^T cov (x-ctr)
struct quadratic_objective {
  vec_type ctr;
  mat_type cov;
  vec_type grad;

  quadratic_objective(const vec_type& ctr, const mat_type& cov)
    : ctr(ctr), cov(cov) { }
  
  double value(const vec_type& x) {
    vec_type diff = x - ctr;
    return 0.5 * dot(diff, cov * diff);
  }

  const vec_type& gradient(const vec_type& x) {
    grad = cov * (x - ctr);
    return grad;
  }
  
}; // struct quadratic_objective


#endif

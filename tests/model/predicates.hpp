#ifndef SILL_TEST_MODEL_PREDICATES_HPP
#define SILL_TEST_MODEL_PREDICATES_HPP

#include <sill/base/finite_assignment_iterator.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/model/interfaces.hpp>

#include <algorithm>
#include <vector>

#include <boost/range/algorithm.hpp>

#include "../predicates.hpp"

using sill::factorized_model;
using sill::table_factor;

struct factor_arg_less {
  template <typename F>
  bool operator()(const F& a, const F& b) {
    return boost::lexicographical_compare(a.arguments(), b.arguments());
  }
};

template <typename F>
boost::test_tools::predicate_result
model_equal_factors(const factorized_model<F>& a,
                    const factorized_model<F>& b) {
  std::vector<F> a_factors(a.factors().begin(), a.factors().end());
  std::vector<F> b_factors(b.factors().begin(), b.factors().end());
  std::sort(a_factors.begin(), a_factors.end(), factor_arg_less());
  std::sort(b_factors.begin(), b_factors.end(), factor_arg_less());
  if (a_factors != b_factors) {
    boost::test_tools::predicate_result result(false);
    result.message() << "The two models do not have equivalent factor sets:\n"
                     << a_factors << " != " << b_factors;
    return result;
  }
  return true;
}

boost::test_tools::predicate_result
model_close_log_likelihoods(const factorized_model<table_factor>& a,
                            const factorized_model<table_factor>& b,
                            double eps) {
  if (a.arguments() != b.arguments()) {
    boost::test_tools::predicate_result result(false);
    result.message() << "The two models do not have identical argument sets: "
                     << a.arguments() << " != " << b.arguments();
    return result;
  }
  sill::finite_assignment_iterator it(a.arguments()), end;
  for(; it != end; ++it) {
    if (std::abs(a.log_likelihood(*it) - b.log_likelihood(*it)) > eps) {
      boost::test_tools::predicate_result result(false);
      result.message() << "The two models differ on the assignment\n"
                       << (*it)
                       << "[" << a.log_likelihood(*it) 
                       << "," << b.log_likelihood(*it)
                       << "]";
      return result;
    }
  }
  return true;
}


#endif


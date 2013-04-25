#ifndef SILL_TEST_FACTOR_PREDICATES_HPP
#define SILL_TEST_FACTOR_PREDICATES_HPP

#include <fstream>
#include <cstdio>

#include <sill/factor/canonical_gaussian.hpp>

#include "../predicates.hpp"

//! Verifies that two factors are close enough
template <typename F>
boost::test_tools::predicate_result
are_close(const F& a, const F& b, typename F::result_type eps) {
  typename F::result_type norma = a.norm_constant();
  typename F::result_type normb = b.norm_constant();
  if (a.arguments() == b.arguments() &&
      norm_inf(a, b) < eps &&
      (norma > normb ? norma - normb : normb - norma) < eps) {
     return true;
  } else {
    boost::test_tools::predicate_result result(false);
    result.message() << "the two factors differ [\n"
                     << a << "!=" << b << "]";
    return result;
  }
}

template <>
boost::test_tools::predicate_result
are_close(const sill::canonical_gaussian& a,
          const sill::canonical_gaussian& b,
          sill::logarithmic<double> eps) {
  double multa = a.log_multiplier();
  double multb = b.log_multiplier();
  if (a.arguments() == b.arguments() &&
      norm_inf(a, b) < eps &&
      (multa > multb ? multa - multb : multb - multa) < eps) {
     return true;
  } else {
    boost::test_tools::predicate_result result(false);
    result.message() << "the two factors differ [\n"
                     << a << "!=" << b << "]";
    return result;
  }
}

#endif

#ifndef SILL_TEST_FACTOR_PREDICATES_HPP
#define SILL_TEST_FACTOR_PREDICATES_HPP

#include <fstream>
#include <cstdio>

#include <sill/factor/canonical_gaussian.hpp>

#include "../predicates.hpp"

// Checks the basic properties of finite table (and matrix) factors
template <typename F>
boost::test_tools::predicate_result
table_properties(const F& f, const typename F::var_vector_type& vars) {
  size_t n = 1;
  for (sill::finite_variable* v : vars) { n *= v->size(); }

  if (f.empty()) {
    boost::test_tools::predicate_result result(false);
    result.message() << "The factor is empty [" << f << "]";
    return result;
  }
  if (f.arity() != vars.size()) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid factor arity ["
                     << f.arity() << " != " << vars.size() << "]";
    return result;
  }
  if (f.size() != n) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid factor size ["
                     << f.size() << " != " << n << "]";
    return result;
  }
  if (f.arguments() != make_domain(vars)) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid factor domain ["
                     << f.arguments() << " != " << make_domain(vars) << "]";
    return result;
  }
  if (f.arg_vector() != vars) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid factor argument vector ["
                     << f.arg_vector() << " != " << vars << "]";
    return result;
  }
  return true;
}

// Verifies that two factors are close enough
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


#include <sill/learning/learn_factor.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  template <>
  table_factor
  learn_marginal<table_factor>(const finite_domain& X, const dataset& ds,
                               double smoothing) {
    assert(includes(ds.finite_variables(), X));
    assert(smoothing >= 0);
    assert(ds.size() > 0);

    table_factor factor(X, smoothing);

    size_t i(0);
    dataset::assignment_iterator end_it(ds.end_assignments());
    for (dataset::assignment_iterator a_it(ds.begin_assignments());
         a_it != end_it; ++a_it) {
      const finite_assignment& a = *a_it;
      factor.v(a) += ds.weight(i);
      ++i;
    }

    factor.normalize();
    return factor;
  }

  template <>
  table_factor
  learn_conditional<table_factor>(const finite_domain& A,
                                  const finite_domain& B,
                                  const finite_assignment& c, const dataset& ds,
                                  double smoothing) {
    assert(includes(ds.finite_variables(), A));
    assert(includes(ds.finite_variables(), B));
    finite_domain Cvars(keys(c));
    assert(includes(ds.finite_variables(), Cvars));
    finite_domain ABvars(set_union(A,B));
    assert(ABvars.size() == A.size() + B.size());
    assert(set_disjoint(ABvars, Cvars));
    assert(smoothing >= 0);
    assert(ds.size() > 0);
    if (A.empty())
      return table_factor(1);

    // Create a constant factor over A,B
    table_factor factor(ABvars, smoothing);

    size_t i(0);
    double total_weight(0);
    dataset::assignment_iterator end_it(ds.end_assignments());
    for (dataset::assignment_iterator a_it(ds.begin_assignments());
         a_it != end_it; ++a_it) {
      const finite_assignment& fa = a_it->finite();
      bool matches_c(true);
      for (finite_assignment::const_iterator c_it(c.begin());
           c_it != c.end(); ++c_it) {
        finite_assignment::const_iterator fa_it(fa.find(c_it->first));
        assert(fa_it != fa.end());
        if (fa_it->second != c_it->second) {
          matches_c = false;
          break;
        }
      }
      total_weight += ds.weight(i);
      if (!matches_c)
        continue;
      factor.v(fa) += ds.weight(i);
      ++i;
    }
    factor.normalize();
    factor = factor.conditional(B);

    return factor;
  } // learn_conditional()

  template <>
  moment_gaussian
  learn_marginal<moment_gaussian>(const vector_domain& Xdom, const dataset& ds,
                                  double reg_cov) {
    assert(ds.size() > 0);
    assert(reg_cov >= 0.);
    assert(includes(ds.vector_variables(), Xdom));
    vector_var_vector X(Xdom.begin(), Xdom.end());
    vec mu;
    ds.mean(mu, X);
    mat cov;
    ds.covariance(cov, X);
    if (reg_cov > 0)
      cov += (reg_cov / ds.size()) * identity(X.size());
    return moment_gaussian(X, mu, cov);
  }

  template <>
  canonical_gaussian
  learn_marginal<canonical_gaussian>(const vector_domain& Xdom,
                                     const dataset& ds, double reg_cov) {
    assert(ds.size() > 0);
    assert(reg_cov >= 0.);
    assert(includes(ds.vector_variables(), Xdom));
    vector_var_vector X(Xdom.begin(), Xdom.end());
    vec mu;
    mat cov;
    ds.mean_covariance(mu, cov, X);
    if (reg_cov > 0)
      cov += (reg_cov / ds.size()) * identity(X.size());
    mat lambda;
    bool result = inv(cov, lambda);
    if (!result)
      throw std::runtime_error
        ("Matrix inverse failed in canonical_gaussian::learn_marginal().");
    return canonical_gaussian(X, lambda, lambda * mu);
  }

} // namespace sill

#include <sill/macros_undef.hpp>

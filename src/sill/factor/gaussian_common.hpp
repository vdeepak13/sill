// Code that depends on the definitions of both moment_gaussian and
// canonical_gaussian. Not intended to be included directly.
#ifndef SILL_GAUSSIAN_COMMON_HPP
#define SILL_GAUSSIAN_COMMON_HPP

#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/moment_gaussian.hpp>
#include <sill/factor/factor_evaluator.hpp>
#include <sill/factor/factor_sampler.hpp>

#include <boost/random/normal_distribution.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  template <typename RandomNumberGenerator>
  vector_assignment
  canonical_gaussian::sample(RandomNumberGenerator& rng) const {
    moment_gaussian mg(*this);
    return mg.sample(rng);
  }

  //! Specialization of factor_evaluator for moment Gaussians
  template <>
  class factor_evaluator<moment_gaussian> {
  public:
    typedef vec                 index_type;
    typedef logarithmic<double> result_type;
    typedef vector_var_vector   arg_vector_type;
    
    factor_evaluator(const moment_gaussian& mg)
      : cg(mg) { }

    logarithmic<double> operator()(const vec& arg) const {
      return cg(arg);
    }
    
    const vector_var_vector& arg_vector() const {
      return cg.arg_vector();
    }

  private:
    const canonical_gaussian cg;
  }; // class factor_evaluator<moment_gaussian>


  //! Specialization of factor_sampler for moment Gaussians
  template <>
  class factor_sampler<moment_gaussian> {
  public:
    typedef vec               index_type;
    typedef vector_var_vector var_vector_type;

    //! Creates a sampler for a marginal distribution
    factor_sampler(const moment_gaussian& factor)
      : coeff_(factor.coefficients()),
        cmean_(factor.mean()),
        trans_(chol(factor.covariance())) {
      assert(factor.marginal());
    }

    //! Creates a sampler for a conditional distribution
    factor_sampler(const moment_gaussian& factor,
                   const vector_var_vector& head)
      : coeff_(factor.coefficients()),
        cmean_(factor.mean()),
        trans_(chol(factor.covariance())) {
      assert(factor.head() == head);
    }

    //! Draws a random sample from a marginal distribution
    template <typename RandomNumberGenerator>
    void operator()(vec& sample, RandomNumberGenerator& rng) const {
      assert(coeff_.empty());
      boost::normal_distribution<> normal;
      vec vals(trans_.n_rows);
      foreach(double& val, vals) { val = normal(rng); }
      sample = cmean_ + trans_.t() * vals;
    }

    //! Draws a random sample from a conditional distribution
    template <typename RandomNumberGenerator>
    void operator()(vec& sample, const vec& tail,
                    RandomNumberGenerator& rng) const {
      boost::normal_distribution<> normal;
      vec vals(trans_.n_rows);
      foreach(double& val, vals) { val = normal(rng); }
      sample = cmean_ + coeff_ * tail + trans_.t() * vals;
    }

  private:
    mat coeff_;
    vec cmean_;
    mat trans_;

  }; // class factor_sampelr<moment_gaussian>

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

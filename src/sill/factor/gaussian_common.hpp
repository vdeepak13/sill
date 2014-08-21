// Code that depends on the definitions of both moment_gaussian and
// canonical_gaussian. Not intended to be included directly.
#ifndef SILL_GAUSSIAN_COMMON_HPP
#define SILL_GAUSSIAN_COMMON_HPP

#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/moment_gaussian.hpp>
#include <sill/factor/factor_evaluator.hpp>

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

} // namespace sill

#endif

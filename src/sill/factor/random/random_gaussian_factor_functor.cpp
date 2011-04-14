
#include <sill/factor/random/random_gaussian_factor_functor.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  template <>
  moment_gaussian
  random_gaussian_factor_functor<moment_gaussian>::
  generate_marginal(const domain_type& X) {
    return generate_marginal_mg(X);
  }

  template <>
  moment_gaussian
  random_gaussian_factor_functor<moment_gaussian>::
  generate_conditional(const domain_type& Y, const domain_type& X) {
    return generate_conditional_mg(Y,X);
  }

} // namespace sill

#include <sill/macros_undef.hpp>

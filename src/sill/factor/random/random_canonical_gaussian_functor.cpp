
#include <sill/factor/random/random_canonical_gaussian_functor.hpp>

namespace sill {

  random_canonical_gaussian_functor::
  random_canonical_gaussian_functor
  (const random_moment_gaussian_functor& rmg_func)
    : rmg_func(rmg_func) { }

  canonical_gaussian
  random_canonical_gaussian_functor::
  generate_marginal(const domain_type& X) {
    return canonical_gaussian(rmg_func.generate_marginal(X));
  }

  canonical_gaussian
  random_canonical_gaussian_functor::
  generate_conditional(const domain_type& Y, const domain_type& X) {
    return canonical_gaussian(rmg_func.generate_conditional(Y, X));
  }

  vector_variable*
  random_canonical_gaussian_functor::
  generate_variable(universe& u, const std::string& name) const {
    return rmg_func.generate_variable(u, name);
  }

  void random_canonical_gaussian_functor::seed(unsigned random_seed) {
    rmg_func.seed(random_seed);
  }

} // namespace sill


#include <sill/factor/random/random_gaussian_crf_factor_functor.hpp>

namespace sill {

  random_gaussian_crf_factor_functor::
  random_gaussian_crf_factor_functor(unsigned random_seed) {
    seed(random_seed);
  }

  random_gaussian_crf_factor_functor::
  random_gaussian_crf_factor_functor
  (const random_moment_gaussian_functor& rmg_func) {
    params.rmg_func = rmg_func;
  }

  gaussian_crf_factor
  random_gaussian_crf_factor_functor::
  generate_marginal(const output_domain_type& Y) {
    return crf_factor_type(params.rmg_func.generate_marginal(Y),
                           Y, input_domain_type());
  }

  gaussian_crf_factor
  random_gaussian_crf_factor_functor::
  generate_conditional(const output_domain_type& Y,
                       const input_domain_type& X) {
    return crf_factor_type(params.rmg_func.generate_conditional(Y,X),
                           Y, X);
  }

  vector_variable*
  random_gaussian_crf_factor_functor::
  generate_output_variable(universe& u, const std::string& name) const {
    return u.new_vector_variable(name, 1);
  }

  vector_variable*
  random_gaussian_crf_factor_functor::
  generate_input_variable(universe& u, const std::string& name) const {
    return u.new_vector_variable(name, 1);
  }

  void
  random_gaussian_crf_factor_functor::seed(unsigned random_seed) {
    params.rmg_func.seed(random_seed);
  }

} // namespace sill

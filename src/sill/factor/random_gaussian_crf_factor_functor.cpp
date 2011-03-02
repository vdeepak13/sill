
#include <sill/factor/random_gaussian_crf_factor_functor.hpp>

namespace sill {

  random_gaussian_crf_factor_functor::random_gaussian_crf_factor_functor
  (random_factor_functor<moment_gaussian>& mg_factor_func)
    : mg_factor_func_ptr(&mg_factor_func), cg_factor_func_ptr(NULL) { }

  random_gaussian_crf_factor_functor::random_gaussian_crf_factor_functor
  (random_factor_functor<canonical_gaussian>& cg_factor_func)
    : mg_factor_func_ptr(NULL), cg_factor_func_ptr(&cg_factor_func) { }

  gaussian_crf_factor
  random_gaussian_crf_factor_functor::
  generate_marginal(const output_domain_type& Y) {
    if (mg_factor_func_ptr) {
      crf_factor_type
        f(moment_gaussian(mg_factor_func_ptr->generate_marginal(Y)),
          Y, input_domain_type());
      return f;
    } else if (cg_factor_func_ptr) {
      crf_factor_type f(cg_factor_func_ptr->generate_marginal(Y),
                        Y, input_domain_type());
      return f;
    } else {
      throw std::runtime_error
        (std::string("random_gaussian_crf_factor_functor::generate_marginal") +
         " (default version) called, but the functor was not constructed" +
         " with a random_factor_functor.");
    }
  }

  gaussian_crf_factor
  random_gaussian_crf_factor_functor::
  generate_conditional(const output_domain_type& Y,
                       const input_domain_type& X) {
    if (mg_factor_func_ptr) {
      crf_factor_type f(mg_factor_func_ptr->generate_conditional(Y,X), Y, X);
      return f;
    } else if (cg_factor_func_ptr) {
      crf_factor_type f(cg_factor_func_ptr->generate_conditional(Y,X), Y, X);
      return f;
    } else {
      throw std::runtime_error
        (std::string("random_gaussian_crf_factor_functor::generate_conditional")
         + " (default version) called, but the functor was not constructed" +
         " with a random_factor_functor.");
    }
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

} // namespace sill

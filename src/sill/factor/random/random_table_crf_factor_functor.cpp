
#include <sill/factor/random/random_table_crf_factor_functor.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  random_table_crf_factor_functor::
  random_table_crf_factor_functor(unsigned random_seed) {
    seed(random_seed);
  }

  table_crf_factor
  random_table_crf_factor_functor::
  generate_marginal(const finite_domain& Y) {
    table_crf_factor f(params.table_factor_func.generate_marginal(Y), Y, false);
    f.convert_to_log_space();
    return f;
  }

  table_crf_factor
  random_table_crf_factor_functor::
  generate_conditional(const finite_domain& Y, const finite_domain& X) {
    table_crf_factor
      f(params.table_factor_func.generate_conditional(Y,X), Y, false);
    f.convert_to_log_space();
    return f;
  }

  finite_variable*
  random_table_crf_factor_functor::
  generate_output_variable(universe& u, const std::string& name) const {
    return params.table_factor_func.generate_variable(u, name);
  }

  finite_variable*
  random_table_crf_factor_functor::
  generate_input_variable(universe& u, const std::string& name) const {
    return params.table_factor_func.generate_variable(u, name);
  }

  void
  random_table_crf_factor_functor::seed(unsigned random_seed) {
    params.table_factor_func.seed(random_seed);
  }

} // namespace sill

#include <sill/macros_undef.hpp>

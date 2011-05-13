
#include <sill/optimization/stochastic_gradient.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  stochastic_gradient_parameters::stochastic_gradient_parameters()
    : base() {
    set_defaults();
  }

  stochastic_gradient_parameters::
  stochastic_gradient_parameters
  (const gradient_method_parameters& gm_params)
    : base(gm_params) {
    set_defaults();
  }

  void stochastic_gradient_parameters::set_defaults() {
    step_type = SINGLE_OPT_STEP;
    single_opt_step_params.eta_choice =
      single_opt_step_parameters::DECREASING_ETA;
    single_opt_step_params.init_eta = 1;
    single_opt_step_params.set_shrink_eta(10000);
  }

  bool stochastic_gradient_parameters::valid() const {
    if (step_type != SINGLE_OPT_STEP)
      return false;
    return gradient_method_parameters::valid();
  }

  void stochastic_gradient_parameters::print(std::ostream& out) const {
    base::print(out);
  }

  std::ostream&
  operator<<(std::ostream& out,
             const stochastic_gradient_parameters& params) {
    params.print(out);
    return out;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

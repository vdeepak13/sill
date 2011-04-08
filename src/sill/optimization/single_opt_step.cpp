
#include <sill/optimization/single_opt_step.hpp>

namespace sill {

  void single_opt_step_parameters::save(oarchive& ar) const {
    ar << eta_choice << fixed_eta;
  }

  void single_opt_step_parameters::load(iarchive& ar) {
    ar >> eta_choice >> fixed_eta;
  }

  void single_opt_step_parameters::
  print(std::ostream& out, const std::string& line_prefix) const {
    out << line_prefix << "eta_choice: " << eta_choice << "\n"
        << line_prefix << "fixed_eta: " << fixed_eta << "\n";
  }

  oarchive&
  operator<<(oarchive& a,
             single_opt_step_parameters::eta_choice_enum eta_choice) {
    a << (size_t)(eta_choice);
    return a;
  }

  iarchive&
  operator>>(iarchive& a,
             single_opt_step_parameters::eta_choice_enum& eta_choice) {
    size_t tmp;
    a >> tmp;
    eta_choice = (single_opt_step_parameters::eta_choice_enum)(tmp);
    return a;
  }

} // namespace sill

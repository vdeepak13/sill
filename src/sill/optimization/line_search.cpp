#include <sill/optimization/line_search.hpp>

namespace sill {

  line_search_parameters::line_search_parameters()
    : convergence_zero(.000001), ls_eta_zero_multiplier(.0000001),
      ls_step_magnitude(1), ls_init_eta(1), ls_eta_mult(2),
      debug(0) { }

  bool line_search_parameters::valid() const {
    if (convergence_zero < 0)
      return false;
    if (ls_eta_zero_multiplier < 0)
      return false;
    if (ls_step_magnitude <= 0)
      return false;
    if (ls_init_eta <= 0)
      return false;
    if (ls_eta_mult <= 1)
      return false;
    return true;
  }

  void line_search_parameters::
  print(std::ostream& out, const std::string& line_prefix) const {
    out << line_prefix << "convergence_zero: " << convergence_zero << "\n"
        << line_prefix << "ls_eta_zero_multiplier: " << ls_eta_zero_multiplier
        << "\n"
        << line_prefix << "ls_step_magnitude: " << ls_step_magnitude << "\n"
        << line_prefix << "ls_init_eta: " << ls_init_eta << "\n"
        << line_prefix << "ls_eta_mult: " << ls_eta_mult << "\n";
  }

  void line_search_parameters::save(oarchive& ar) const {
    ar << convergence_zero << ls_eta_zero_multiplier << ls_step_magnitude
       << ls_init_eta << ls_eta_mult << debug;
  }

  void line_search_parameters::load(iarchive& ar) {
    ar >> convergence_zero >> ls_eta_zero_multiplier >> ls_step_magnitude
       >> ls_init_eta >> ls_eta_mult >> debug;
  }

} // namespace sill

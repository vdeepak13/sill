#include <sill/optimization/gradient_method.hpp>

namespace sill {

  bool gradient_method_parameters::valid(bool print_warnings) const {
    switch (step_type) {
    case SINGLE_OPT_STEP:
      if (!single_opt_step_params.valid())
        return false;
      break;
    case LINE_SEARCH:
    case LINE_SEARCH_WITH_GRAD:
      if (!ls_params.valid())
        return false;
      break;
    default:
      assert(false);
    }
    if (convergence_zero < 0)
      return false;
    return true;
  }

  void gradient_method_parameters::save(oarchive& ar) const {
    ar << step_type << ls_params << single_opt_step_params << ls_stopping
       << convergence_zero << debug;
  }

  void gradient_method_parameters::load(iarchive& ar) {
    ar >> step_type >> ls_params >> single_opt_step_params >> ls_stopping
       >> convergence_zero >> debug;
  }

  void gradient_method_parameters::
  print(std::ostream& out, const std::string& line_prefix) const {
    out << line_prefix << "step_type: " << step_type << "\n"
        << line_prefix << "ls_params:\n";
    ls_params.print(out, line_prefix + "  ");
    out << line_prefix << "single_opt_step_params:\n";
    single_opt_step_params.print(out, line_prefix + "  ");
    out << line_prefix << "ls_stopping: " << ls_stopping << "\n"
        << line_prefix << "convergence_zero: " << convergence_zero << "\n"
        << line_prefix << "debug: " << debug << "\n";
  }

  oarchive&
  operator<<(oarchive& a,
             gradient_method_parameters::real_opt_step_type val) {
    a << (size_t)(val);
    return a;
  }

  iarchive&
  operator>>(iarchive& a,
             gradient_method_parameters::real_opt_step_type& val) {
    size_t tmp;
    a >> tmp;
    val = (gradient_method_parameters::real_opt_step_type)(tmp);
    return a;
  }

  oarchive&
  operator<<(oarchive& a,
             gradient_method_parameters::ls_stopping_type val) {
    a << (size_t)(val);
    return a;
  }

  iarchive&
  operator>>(iarchive& a,
             gradient_method_parameters::ls_stopping_type& val) {
    size_t tmp;
    a >> tmp;
    val = (gradient_method_parameters::ls_stopping_type)(tmp);
    return a;
  }

  std::ostream&
  operator<<(std::ostream& out, const gradient_method_parameters& gm_params) {
    gm_params.print(out);
    return out;
  }

}; // namespace sill

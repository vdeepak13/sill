
#include <sill/optimization/single_opt_step_builder.hpp>

namespace sill {

  void single_opt_step_builder::add_options
  (boost::program_options::options_description& desc,
   const std::string& desc_prefix) {

    namespace po = boost::program_options;
    po::options_description
      sub_desc1(desc_prefix + "Single Optimization Step Options");
    sub_desc1.add_options()
      ("eta_choice",
       po::value<std::string>(&eta_choice_string)->default_value("fixed_eta"),
       "Method for choosing step size eta.")
      ("init_eta",
       po::value<double>(&(sos_params.init_eta))->default_value(.1),
       "Initial step size value (> 0).")
      ("shrink_eta",
       po::value<double>(&(sos_params.shrink_eta))->default_value(.999),
       "(For DECREASING_ETA) Discount factor in (0,1] by which eta is shrunk each round.");
    desc.add(sub_desc1);

  } // add_options

  const single_opt_step_parameters& single_opt_step_builder::get_parameters() {
    if (eta_choice_string == "fixed_eta") {
      sos_params.eta_choice = single_opt_step_parameters::FIXED_ETA;
    } else {
      throw std::invalid_argument
        ("Bad value for eta_choice: " + eta_choice_string);
    }
    return sos_params;
  }

}; // namespace sill

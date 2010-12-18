
#include <sill/optimization/line_search_builder.hpp>

namespace sill {

  void line_search_builder::add_options
  (boost::program_options::options_description& desc,
   const std::string& desc_prefix) {

    namespace po = boost::program_options;
    po::options_description
      sub_desc1(desc_prefix + "Line Search Options");
    sub_desc1.add_options()
      ("ls_eta_zero_multiplier",
       po::value<double>(&(ls_params.ls_eta_zero_multiplier))
       ->default_value(.0000001),
       "If the step size eta (times the ls_step_magnitude option) is less than ls_eta_zero_multiplier * convergence_zero, then the line search will declare convergence.")
      ("ls_step_magnitude",
       po::value<double>(&(ls_params.ls_step_magnitude))->default_value(1),
       "The magnitude of the step size when eta == 1; this allows the search to determine convergence in terms of the absolute value of the actual step size, rather than in terms of the multiplier eta.")
      ("ls_init_eta",
       po::value<double>(&(ls_params.ls_init_eta))->default_value(1),
       "Initial step size multiplier to try.")
      ("ls_eta_mult",
       po::value<double>(&(ls_params.ls_eta_mult))->default_value(2),
       "Value (> 1) by which the step size multiplier eta is multiplied/divided by on each step of the search.");
    const po::option_description* find_option_ptr =
      desc.find_nothrow("convergence_zero", false);
    if (!find_option_ptr) {
      sub_desc1.add_options()
        ("convergence_zero",
         po::value<double>(&(ls_params.convergence_zero))
         ->default_value(.000001),
         "Convergence tolerance.");
    }
    find_option_ptr = desc.find_nothrow("debug", false);
    if (!find_option_ptr) {
      sub_desc1.add_options()
        ("debug",
         po::value<size_t>(&(ls_params.debug))->default_value(0),
         "Print debugging info for values > 0.");
    }
    desc.add(sub_desc1);

  } // add_options

} // end of namespace: prl

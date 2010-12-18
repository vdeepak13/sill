
#include <sill/optimization/gradient_method_builder.hpp>

namespace sill {

  void gradient_method_builder::add_options
  (boost::program_options::options_description& desc,
   const std::string& desc_prefix) {
    namespace po = boost::program_options;
    po::options_description
      sub_desc1(desc_prefix + "Gradient Method Options");
    sub_desc1.add_options()
      ("step_type",
       po::value<std::string>(&step_type_string)
       ->default_value("line_search"),
       "Optimization step type (single_opt_step, line_search, line_search_with_grad)")
      ("ls_stopping",
       po::value<std::string>(&ls_stopping_string)
       ->default_value("ls_exact"),
       "Line search stopping conditions (ls_exact, ls_strong_wolfe, ls_weak_wolfe)");
    const po::option_description* find_option_ptr =
      desc.find_nothrow("convergence_zero", false);
    if (!find_option_ptr) {
      sub_desc1.add_options()
        ("convergence_zero",
         po::value<double>(&(gm_params.convergence_zero))
         ->default_value(.000001),
         "Convergence tolerance.");
    }
    find_option_ptr = desc.find_nothrow("debug", false);
    if (!find_option_ptr) {
      sub_desc1.add_options()
        ("debug",
         po::value<size_t>(&(gm_params.debug))->default_value(0),
         "Print debugging info for values > 0.");
    }
    desc.add(sub_desc1);
    ls_builder.add_options(desc, desc_prefix + "Gradient Method: ");
    sos_builder.add_options(desc, desc_prefix + "Gradient Method: ");
  } // add_options

  const gradient_method_parameters& gradient_method_builder::get_parameters() {
    if (step_type_string == "single_opt_step") {
      gm_params.step_type = gradient_method_parameters::SINGLE_OPT_STEP;
    } else if (step_type_string == "line_search") {
      gm_params.step_type = gradient_method_parameters::LINE_SEARCH;
    } else if (step_type_string == "line_search_with_grad") {
      gm_params.step_type = gradient_method_parameters::LINE_SEARCH_WITH_GRAD;
    } else {
      throw std::invalid_argument
        ("gradient_method_builder given invalid step_type: "
         + step_type_string);
    }
    gm_params.ls_params = ls_builder.get_parameters();
    gm_params.ls_params.convergence_zero = gm_params.convergence_zero;
    gm_params.ls_params.debug = (gm_params.debug > 0 ?
                                 gm_params.debug - 1 :
                                 0);
    gm_params.single_opt_step_params = sos_builder.get_parameters();
    if (ls_stopping_string == "ls_exact") {
      gm_params.ls_stopping = gradient_method_parameters::LS_EXACT;
    } else if (ls_stopping_string == "ls_strong_wolfe") {
      gm_params.ls_stopping = gradient_method_parameters::LS_STRONG_WOLFE;
    } else if (ls_stopping_string == "ls_weak_wolfe") {
      gm_params.ls_stopping = gradient_method_parameters::LS_WEAK_WOLFE;
    } else {
      throw std::invalid_argument
        ("gradient_method_builder given invalid ls_stopping value: "
         + ls_stopping_string);
    }
    return gm_params;
  } // get_parameters

}; // namespace sill

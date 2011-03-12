
#include <sill/optimization/real_optimizer_builder.hpp>

namespace sill {

  bool real_optimizer_builder::is_stochastic(real_optimizer_type rot) {
    switch (rot) {
    case GRADIENT_DESCENT:
    case CONJUGATE_GRADIENT:
    case CONJUGATE_GRADIENT_DIAG_PREC:
    case LBFGS:
      return false;
    case STOCHASTIC_GRADIENT:
      return true;
    default:
      assert(false);
      return false;
    }
  }

  real_optimizer_builder::real_optimizer_type
  real_optimizer_builder::method() const {
    if (method_string == "gradient_descent") {
      return GRADIENT_DESCENT;
    } else if (method_string == "conjugate_gradient") {
      return CONJUGATE_GRADIENT;
    } else if (method_string == "conjugate_gradient_diag_prec") {
      return CONJUGATE_GRADIENT_DIAG_PREC;
    } else if (method_string == "lbfgs") {
      return LBFGS;
    } else if (method_string == "stochastic_gradient") {
      return STOCHASTIC_GRADIENT;
    } else {
      throw std::invalid_argument
        ("real_optimizer_builder given invalid method: " + method_string);
    }
  }

  void real_optimizer_builder::add_options
  (boost::program_options::options_description& desc,
   const std::string& desc_prefix) {

    namespace po = boost::program_options;
    po::options_description
      sub_desc1(desc_prefix + "Real-Valued Optimization Options");
    sub_desc1.add_options()
      ("method",
       po::value<std::string>(&method_string)
       ->default_value("conjugate_gradient"),
       "Optimization method (gradient_descent, conjugate_gradient, diag_prec_conjugate_gradient, lbfgs, stochastic_gradient).")
      ("cg_update_method",
       po::value<size_t>(&cg_update_method)->default_value(0),
       "(For CONJUGATE_GRADIENT*) Update method. 0: beta = max{0, Polak-Ribiere}")
      ("lbfgs_M",
       po::value<size_t>(&lbfgs_M)->default_value(10),
       "(For LBFGS) Save M (> 0) previous gradients for estimating the Hessian.");
    desc.add(sub_desc1);
    gm_builder.add_options
      (desc, desc_prefix + "Real-Valued Optimization: ");
  }

  gradient_descent_parameters real_optimizer_builder::get_gd_parameters() {
    gradient_descent_parameters params(gm_builder.get_parameters());
    return params;
  }

  conjugate_gradient_parameters real_optimizer_builder::get_cg_parameters() {
    conjugate_gradient_parameters params(gm_builder.get_parameters());
    params.update_method = cg_update_method;
    return params;
  }

  lbfgs_parameters real_optimizer_builder::get_lbfgs_parameters() {
    lbfgs_parameters params(gm_builder.get_parameters());
    params.M = lbfgs_M;
    return params;
  }

  stochastic_gradient_parameters real_optimizer_builder::get_sg_parameters() {
    throw std::runtime_error("NOT YET IMPLEMENTED!");
    stochastic_gradient_parameters params;
    return params;
  }

}; // namespace sill

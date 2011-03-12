#include <limits>

#include <sill/learning/crossval_methods.hpp>
#include <sill/learning/dataset/dataset_view.hpp>
#include <sill/learning/dataset/record_conversions.hpp>
#include <sill/learning/discriminative/multiclass_logistic_regression.hpp>
#include <sill/learning/validation/validation_framework.hpp>
#include <sill/math/permutations.hpp>
#include <sill/stl_io.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // multiclass_logistic_regression_builder
  //==========================================================================

  void multiclass_logistic_regression_builder::
  add_options(boost::program_options::options_description& desc,
              const std::string& desc_prefix) {
    namespace po = boost::program_options;

    po::options_description
      sub_desc1(desc_prefix + "Multiclass LogReg Options: Learning");
    sub_desc1.add_options()
      ("regularization",
       po::value<size_t>(&(mlr_params.regularization))->default_value(2),
       "Regularization: 0 (none), 2 (L2).")
      ("lambda",
       po::value<double>(&(mlr_params.lambda))->default_value(.00001),
       "Regularization parameter.")
      ("init_iterations",
       po::value<size_t>(&(mlr_params.init_iterations))->default_value(1000),
       "Number of initial iterations to run.")
      ("perturb",
       po::value<double>(&(mlr_params.perturb))->default_value(0),
       "Amount of perturbation (Uniform[-perturb,perturb]) to use in choosing initial weights. (>= 0)");
    desc.add(sub_desc1);

    po::options_description
      sub_desc2(std::string("Multiclass LogReg Options: Other"));
    sub_desc2.add_options()
      ("resolve_numerical_problems",
       po::bool_switch(&(mlr_params.resolve_numerical_problems)),
       "Heuristically resolve numerical problems from large weights. (default = false)");
    const po::option_description* find_option_ptr =
      desc.find_nothrow("random_seed", false);
    if (!find_option_ptr) {
      sub_desc2.add_options()
        ("random_seed",
         po::value<unsigned>(&(mlr_params.random_seed))
         ->default_value(time(NULL)),
         "Random seed (default = time).");
    }
    find_option_ptr = desc.find_nothrow("debug", false);
    if (!find_option_ptr) {
      sub_desc2.add_options()
        ("debug",
         po::value<size_t>(&(mlr_params.debug))->default_value(0),
         "Print debugging info for values > 0.");
    }
    desc.add(sub_desc2);

    real_opt_builder.add_options(desc, "Multiclass LogReg: ");
  } // add_options

  const multiclass_logistic_regression_parameters&
  multiclass_logistic_regression_builder::get_parameters() {
    mlr_params.opt_method = real_opt_builder.method();
    mlr_params.gm_params = real_opt_builder.get_gd_parameters();
    mlr_params.gm_params.debug = (mlr_params.debug > 0 ?
                                  mlr_params.debug - 1 :
                                  0);
    mlr_params.cg_update_method =
      real_opt_builder.get_cg_parameters().update_method;
    mlr_params.lbfgs_M = real_opt_builder.get_lbfgs_parameters().M;
    return mlr_params;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

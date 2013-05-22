
#include <sill/learning/crf/crf_parameter_learner_builder.hpp>

namespace sill {

  void crf_parameter_learner_builder::add_options
  (boost::program_options::options_description& desc,
   const std::string& desc_prefix) {

    namespace po = boost::program_options;

    po::options_description
      sub_desc1(desc_prefix + "CRF Parameter Learner: Learning Options");
    sub_desc1.add_options()
      ("regularization",
       po::value<size_t>(&(cpl_params.regularization))->default_value(2),
       "Regularization: 0 (none), 2 (L2).")
      ("lambdas",
       po::value<vec>(&(cpl_params.lambdas))->default_value(zeros<vec>(1)),
       "Regularization parameters (whose meaning depends on the factor type).")
      ("init_iterations",
       po::value<size_t>(&(cpl_params.init_iterations))->default_value(10000),
       "Number of initial iterations of parameter learning to run.")
      ("init_time_limit",
       po::value<size_t>(&(cpl_params.init_time_limit))->default_value(0),
       "Time limit in seconds for initial iterations of parameter learning. If 0, there is no limit.")
      ("learning_objective",
       po::value<std::string>(&learning_objective_string)->default_value("MLE"),
       "Learning objective (MLE = max likelihood; MPLE = max pseudolikelihood)")
      ("perturb",
       po::value<double>(&(cpl_params.perturb))->default_value(0),
       "Amount of perturbation (Uniform[-perturb,perturb]) to use in choosing initial weights for the features. (>= 0)");
    desc.add(sub_desc1);

    po::options_description
      sub_desc2(std::string("CRF Parameter Learner: Other Options"));
    sub_desc2.add_options()
      ("no_shared_computation",
       po::bool_switch(&(cpl_params.no_shared_computation)),
       "If true, do not use the share_computation option in computing the objective, gradient, etc. (default = false)")
      ("keep_fixed_records",
       po::bool_switch(&(cpl_params.keep_fixed_records)),
       "If true, this turns on the fixed_records option for the learned model. (default = false)");
    const po::option_description* find_option_ptr =
      desc.find_nothrow("random_seed", false);
    if (!find_option_ptr) {
      sub_desc2.add_options()
        ("random_seed",
         po::value<unsigned>(&(cpl_params.random_seed))
         ->default_value(time(NULL)),
         "Random seed (default = time).");
    }
    find_option_ptr = desc.find_nothrow("debug", false);
    if (!find_option_ptr) {
      sub_desc2.add_options()
        ("debug",
         po::value<size_t>(&(cpl_params.debug))->default_value(0),
         "Print debugging info for values > 0.");
    }
    desc.add(sub_desc2);

    real_opt_builder.add_options(desc, "CRF Parameter Learner: ");
  } // add_options

  const crf_parameter_learner_parameters&
  crf_parameter_learner_builder::get_parameters() {
    cpl_params.learning_objective = learning_objective();
    cpl_params.opt_method = real_opt_builder.method();
    cpl_params.gm_params = real_opt_builder.get_gd_parameters();
    cpl_params.gm_params.debug = (cpl_params.debug > 0 ?
                                  cpl_params.debug - 1 :
                                  0);
    cpl_params.cg_update_method =
      real_opt_builder.get_cg_parameters().update_method;
    cpl_params.lbfgs_M = real_opt_builder.get_lbfgs_parameters().M;
    return cpl_params;
  }

  crf_parameter_learner_parameters::learning_objective_enum
  crf_parameter_learner_builder::learning_objective() const {
    if (learning_objective_string == "MLE") {
      return crf_parameter_learner_parameters::MLE;
    } else if (learning_objective_string == "MPLE") {
      return crf_parameter_learner_parameters::MPLE;
    } else {
      assert(false);
      return crf_parameter_learner_parameters::MLE;
    }
  }

} // namespace sill

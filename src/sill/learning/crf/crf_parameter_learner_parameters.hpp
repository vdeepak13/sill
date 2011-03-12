#ifndef SILL_CRF_PARAMETER_LEARNER_PARAMETERS_HPP
#define SILL_CRF_PARAMETER_LEARNER_PARAMETERS_HPP

#include <sill/optimization/real_optimizer_builder.hpp>

namespace sill {

  /**
   * Parameters for crf_parameter_learner.
   *
   * This allows easy parsing of command-line options via Boost Program Options.
   *
   * Usage: Create your own Options Description desc.
   *        Call this struct's add_options() method with desc to add synthetic
   *        model options to desc.
   *        Parse the command line using the modified options description.
   *        Pass this struct (which now holds the parsed options) to
   *        crf_parameter_learner.
   */
  struct crf_parameter_learner_parameters {

    // Real optimization parameters
    //==========================================================================

    //! Optimization method.
    real_optimizer_builder::real_optimizer_type opt_method;

    //! Gradient method parameters.
    gradient_method_parameters gm_params;

    //! Conjugate gradient update method.
    size_t cg_update_method;

    //! L-BFGS M.
    size_t lbfgs_M;

    // Learning parameters
    //==========================================================================

    /**
     * Regularization type:
     *  - 0: none
     *  - 2: L2 regularization (default)
     */
    size_t regularization;

    /**
     * Regularization parameters; see factor type for details and defaults.
     * In general, these values should be proportional to the number of
     * pseudoexamples to be represented by the regularization.
     *
     * Note: If the factor type requires k lambda values and this vector is of
     *       length 1, then the 1 value is copied for all k lambdas.
     *       Otherwise, if the length of lambdas does not fit the factor type,
     *       an error is thrown.
     *
     * (default = 0)
     */
    vec lambdas;

    /**
     * Number of initial iterations of parameter learning to run.
     *  (default = 0)
     */
    size_t init_iterations;

    /**
     * Time limit in seconds for initial iterations of parameter learning.
     * If 0, there is no limit.
     *  (default = 0)
     */
    size_t init_time_limit;

    /**
     * Amount of perturbation (Uniform[-perturb,perturb]) to use in choosing
     * initial weights for the features.  Should be >= 0.
     *  (default = 0)
     */
    double perturb;

    // Other parameters
    //==========================================================================

    /**
     * Random seed.
     *  (default = time)
     */
    unsigned random_seed;

    //! If true, do not use the share_computation option in computing the
    //! objective, gradient, etc.
    //!  (default = false)
    bool no_shared_computation;

    /**
     * If true, this turns on the fixed_records option for the learned model.
     * Note: You can turn this on yourself once you retrieve the learned model.
     *  (default = false)
     */
    bool keep_fixed_records;

    /**
     * Print debugging info.
     *  - 0: none (default)
     *  - 1: fixed amount per construction; none per iteration
     *  - 2: fixed amount per iteration
     *  - 3: print info whenever objective, gradient, etc. routines are called
     *  - higher: reverts to highest debugging mode
     */
    size_t debug;

    // Methods
    //==========================================================================

    crf_parameter_learner_parameters();

    /**
     * @param  print_warnings
     *         If true, print warnings to STDERR about invalid options.
     *          (default = true)
     *
     * @return true iff the parameters are valid
     */
    bool valid(bool print_warnings = true) const;

  }; // struct crf_parameter_learner_parameters

}  // namespace sill

#endif // SILL_CRF_PARAMETER_LEARNER_PARAMETERS_HPP

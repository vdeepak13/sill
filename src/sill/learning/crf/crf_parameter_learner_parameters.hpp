#ifndef SILL_CRF_PARAMETER_LEARNER_PARAMETERS_HPP
#define SILL_CRF_PARAMETER_LEARNER_PARAMETERS_HPP

#include <sill/optimization/real_optimizer_builder.hpp>
#include <sill/serialization/serialize.hpp>

namespace sill {

  /**
   * Parameters for crf_parameter_learner.
   * @see crf_parameter_learner
   */
  struct crf_parameter_learner_parameters {

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
     * Templated factors: When using a crf_model with templated_crf_factor,
     * regularization penalties are added once per factor instance. This means
     * that templated factors which appear many times in a model are penalized
     * more than templated factors which appear infrequently.
     *
     * (default = 0)
     */
    vec lambdas;

    /**
     * Number of initial iterations of parameter learning to run.
     *  (default = 10000)
     */
    size_t init_iterations;

    /**
     * Time limit in seconds for initial iterations of parameter learning.
     * If 0, there is no limit.
     *  (default = 0)
     */
    size_t init_time_limit;

    /**
     * Objective types:
     *  - MLE: maximum likelihood
     *  - MPLE: maximum pseudolikelihood
     */
    enum learning_objective_enum { MLE, MPLE };

    //! Learning objective.
    //!  (default = MLE)
    learning_objective_enum learning_objective;

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

    //! If true, do not use the share_computation option in computing the
    //! objective, gradient, etc.
    //!  (default = false)
    bool no_shared_computation;

    // Real optimization parameters
    //==========================================================================

    //! Optimization method.
    real_optimizer_builder::real_optimizer_type opt_method;

    //! Gradient method parameters.
    gradient_method_parameters gm_params;

    //! Conjugate gradient update method.
    //!  (default = 0, i.e., beta = max{0, Polak-Ribiere})
    size_t cg_update_method;

    //! L-BFGS M.
    //!  (default = 10)
    size_t lbfgs_M;

    // Methods
    //==========================================================================

    crf_parameter_learner_parameters();

    //! Check validity; assert false if invalid.
    void check() const;

    void save(oarchive & ar) const;

    void load(iarchive & ar);

  }; // struct crf_parameter_learner_parameters

  oarchive&
  operator<<(oarchive& a,
             crf_parameter_learner_parameters::learning_objective_enum val);

  iarchive&
  operator>>(iarchive& a,
             crf_parameter_learner_parameters::learning_objective_enum& val);

}  // namespace sill

#endif // SILL_CRF_PARAMETER_LEARNER_PARAMETERS_HPP

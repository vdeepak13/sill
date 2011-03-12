
#ifndef SILL_REAL_OPTIMIZER_BUILDER_HPP
#define SILL_REAL_OPTIMIZER_BUILDER_HPP

#include <sill/optimization/conjugate_gradient.hpp>
#include <sill/optimization/gradient_descent.hpp>
#include <sill/optimization/gradient_method_builder.hpp>
#include <sill/optimization/lbfgs.hpp>
#include <sill/optimization/stochastic_gradient.hpp>

namespace sill {

  /**
   * Class for parsing command-line options to specify an optimization method
   * for real-valued problems.
   */
  class real_optimizer_builder {

  public:

    /**
     * Optimization methods.
     *  - 0: gradient descent
     *  - 1: conjugate gradient
     *  - 2: conjugate gradient with a diagonal preconditioner
     *  - 3: L-BFGS
     *  - 4: stochastic gradient descent
     */
    enum real_optimizer_type { GRADIENT_DESCENT, CONJUGATE_GRADIENT,
                               CONJUGATE_GRADIENT_DIAG_PREC, LBFGS,
                               STOCHASTIC_GRADIENT };

    //! Indicates whether the optimization method is stochastic
    //! (requires an oracle).
    static bool is_stochastic(real_optimizer_type rot);

  private:

    /**
     * Text string specifying the optimization method:
     *  - gradient_descent
     *  - conjugate_gradient (default)
     *  - conjugate_gradient_diag_prec
     *  - lbfgs
     *  - stochastic_gradient
     */
    std::string method_string;

    //! For gradient-based methods.
    gradient_method_builder gm_builder;

    /**
     * (CONJUGATE_GRADIENT*)
     * Update method:
     *  - 0: beta = max{0, Polak-Ribiere}
     *    (default)
     */
    size_t cg_update_method;

    //! (L-BFGS)
    //! Save M (> 0) previous gradients for estimating the Hessian.
    //!  (default = 10)
    size_t lbfgs_M;

  public:

    real_optimizer_builder()
      : method_string("conjugate_gradient"), cg_update_method(0), lbfgs_M(10) {
    }

    /**
     * Returns the enumeration value specifying the optimization method
     * corresponding to method_string.
     */
    real_optimizer_type method() const;

    /**
     * Add options to the given Options Description.
     * Once the Options Description is used to parse argv, this struct will
     * hold the specified values.
     */
    void add_options(boost::program_options::options_description& desc,
                     const std::string& desc_prefix = "");

    //! Get parameters for GRADIENT_DESCENT.
    //! This works regardless of the specified method.
    gradient_descent_parameters get_gd_parameters();

    //! Get parameters for CONJUGATE_GRADIENT*.
    //! This works regardless of the specified method.
    conjugate_gradient_parameters get_cg_parameters();

    //! Get parameters for LBFGS.
    //! This works regardless of the specified method.
    lbfgs_parameters get_lbfgs_parameters();

    //! Get parameters for STOCHASTIC_GRADIENT.
    //! This works regardless of the specified method.
    stochastic_gradient_parameters get_sg_parameters();

  }; // class real_optimizer_builder

} // end of namespace: prl

#endif // #ifndef SILL_REAL_OPTIMIZER_BUILDER_HPP

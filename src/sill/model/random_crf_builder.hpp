#ifndef SILL_MODEL_RANDOM_CRF_BUILDER
#define SILL_MODEL_RANDOM_CRF_BUILDER

#include <boost/program_options.hpp>

#include <sill/factor/random/random_gaussian_crf_factor_functor_builder.hpp>
#include <sill/factor/random/random_table_crf_factor_functor_builder.hpp>
#include <sill/model/random.hpp>

namespace sill {

  //! \addtogroup model
  //! @{

  /**
   * Struct which holds parameters specifying options for synthetic CRF models
   * to be constructed using the methods in prl/models/random.hpp.
   *
   * This allows easy parsing of command-line options via Boost Program Options.
   *
   * Usage: Create your own Options Description desc.
   *        Call this struct's add_options() method with desc to add synthetic
   *        model options to desc.
   *        Parse the command line using the modified options description.
   *        Use this struct's create_model() method to create the synthetic
   *        model specified by the options.
   */
  struct random_crf_builder {

    // Model parameters
    //==========================================================================

    //! Factor type: table/gaussian
    //! This implicitly specifies the variable type.
    std::string factor_type;

    std::string model_structure;

    size_t model_size;

    bool tractable;

    bool add_cross_factors;

    //! Factor alternation period.
    //! (0 means no alternation; 1 means use alt parameters only)
    size_t factor_alt_period;

    // Factor parameters for discrete variables
    //==========================================================================

    //! Y-Y table factor parameters.
    //! These are used to specify the variable arity.
    random_table_crf_factor_functor_builder YY_rtcff_builder;

    //! Y-X table factor parameters.
    random_table_crf_factor_functor_builder YX_rtcff_builder;

    //! X-X table factor parameters.
    random_table_factor_functor_builder XX_rtff_builder;

    //! Alternative Y-Y table factor parameters.
    random_table_crf_factor_functor_builder alt_YY_rtcff_builder;

    //! Alternative Y-X table factor parameters.
    random_table_crf_factor_functor_builder alt_YX_rtcff_builder;

    //! Alternative X-X table factor parameters.
    random_table_factor_functor_builder alt_XX_rtff_builder;

    // Factor parameters for real variables
    //==========================================================================

    //! Y-Y Gaussian factor parameters.
    random_gaussian_crf_factor_functor_builder YY_rgcff_builder;

    //! Y-X Gaussian factor parameters.
    random_gaussian_crf_factor_functor_builder YX_rgcff_builder;

    //! X-X Gaussian factor parameters.
    random_moment_gaussian_functor_builder XX_rmgf_builder;

    //! Alternative Y-Y Gaussian factor parameters.
    random_gaussian_crf_factor_functor_builder alt_YY_rgcff_builder;

    //! Alternative Y-X Gaussian factor parameters.
    random_gaussian_crf_factor_functor_builder alt_YX_rgcff_builder;

    //! Alternative X-X Gaussian factor parameters.
    random_moment_gaussian_functor_builder alt_XX_rmgf_builder;

    // Methods
    //==========================================================================

    /**
     * Add options for the above model and factor parameters to the given
     * Options Description.
     * Once the Options Description is used to parse argv, this struct will
     * hold the specified values.
     */
    void add_options(boost::program_options::options_description& desc);

    //! Check options.  Assert false if invalid.
    void check() const;

    /**
     * Generate a synthetic CRF using the current options and the given
     * random seed.
     * Note: The parameters MUST specify a discrete-valued model.
     *
     * @param Xmodel        (Return value) Model for P(X).
     *                      Any model previously stored here will be cleared.
     * @param YgivenXmodel  (Return value) Model for P(Y|X).
     *                      Any model previously stored here will be cleared.
     * @param Y_vec         (Return value) Y variables in order.
     * @param X_vec         (Return value) X variables in order.
     * @param Y2X_map       (Return value) Mapping from Y variables to their
     *                      corresponding X variables.
     */
    void create_model
    (decomposable<table_factor>& Xmodel,
     crf_model<table_crf_factor>& YgivenXmodel,
     finite_var_vector& Y_vec, finite_var_vector& X_vec,
     std::map<finite_variable*, copy_ptr<finite_domain> >& Y2X_map,
     universe& u, unsigned int random_seed) const;

    /**
     * Generate a synthetic CRF using the current options and the given
     * random seed.
     * Note: The parameters MUST specify a real-valued model.
     *
     * @param Xmodel        (Return value) Model for P(X).
     *                      Any model previously stored here will be cleared.
     * @param YgivenXmodel  (Return value) Model for P(Y|X).
     *                      Any model previously stored here will be cleared.
     * @param Y_vec         (Return value) Y variables in order.
     * @param X_vec         (Return value) X variables in order.
     * @param Y2X_map       (Return value) Mapping from Y variables to their
     *                      corresponding X variables.
     */
    void create_model
    (decomposable<canonical_gaussian>& Xmodel,
     crf_model<gaussian_crf_factor>& YgivenXmodel,
     vector_var_vector& Y_vec, vector_var_vector& X_vec,
     std::map<vector_variable*, copy_ptr<vector_domain> >& Y2X_map,
     universe& u, unsigned int random_seed) const;

    //! Print the options in this struct.
    void print(std::ostream& out) const;

  }; // struct random_crf_builder

  //! Print the options in the given random_crf_builder.
  std::ostream& operator<<(std::ostream& out, const random_crf_builder& rcb);

  //! @}

} // namespace sill

#endif // #ifndef SILL_MODEL_RANDOM_CRF_BUILDER

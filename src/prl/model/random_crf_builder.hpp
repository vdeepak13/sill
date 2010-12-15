#ifndef PRL_MODEL_RANDOM_CRF_BUILDER
#define PRL_MODEL_RANDOM_CRF_BUILDER

#include <boost/program_options.hpp>

#include <prl/model/random.hpp>

namespace prl {

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

    std::string model_structure;

    std::string variable_type;

    size_t model_size;

    bool tractable;

    bool add_cross_factors;

    // Factor parameters for discrete variables
    //==========================================================================

    size_t variable_arity;

    std::string factor_type;

    double YYstrength;

    double YXstrength;

    double XXstrength;

    double strength_base;

    size_t alternation_period;

    double altYYstrength;

    double altYXstrength;

    double altXXstrength;

    double alt_strength_base;

    // Factor parameters for real variables
    //==========================================================================

    double b_max;

    double c_max;

    double variance;

    double YYcorrelation;

    double YXcorrelation;

    double XXcorrelation;

    // Methods
    //==========================================================================

    /**
     * Add options for the above model and factor parameters to the given
     * Options Description.
     * Once the Options Description is used to parse argv, this struct will
     * hold the specified values.
     */
    void add_options(boost::program_options::options_description& desc);

    /**
     * Check options.  Return true if valid and false if invalid.
     * @param  print_warnings   If true, print warnings to STDERR about invalid
     *                          options.
     *                          (default = true)
     */
    bool valid(bool print_warnings = true) const;

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

    /**
     * Print the options in this struct.
     */
    void print(std::ostream& out) const;

  }; // struct random_crf_builder

  //! Print the options in the given random_crf_builder.
  std::ostream& operator<<(std::ostream& out, const random_crf_builder& rcb);

  //! @}

} // namespace prl

#endif // #ifndef PRL_MODEL_RANDOM_CRF_BUILDER

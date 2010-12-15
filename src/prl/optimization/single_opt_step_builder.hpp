
#ifndef PRL_SINGLE_OPT_STEP_BUILDER_HPP
#define PRL_SINGLE_OPT_STEP_BUILDER_HPP

#include <boost/program_options.hpp>

#include <prl/optimization/single_opt_step.hpp>

namespace prl {

  /**
   * Class for parsing command-line options which specify
   * single_opt_step_parameters.
   */
  class single_opt_step_builder {

    single_opt_step_parameters sos_params;

    std::string eta_choice_string;

  public:

    single_opt_step_builder() { }

    /**
     * Add options to the given Options Description.
     * Once the Options Description is used to parse argv, this struct will
     * hold the specified values.
     */
    void add_options(boost::program_options::options_description& desc,
                     const std::string& desc_prefix = "");

    //! Return the parameters held by this builder class.
    const single_opt_step_parameters& get_parameters();

  }; // class single_opt_step_builder

}; // end of namespace: prl

#endif // #ifndef PRL_SINGLE_OPT_STEP_BUILDER_HPP

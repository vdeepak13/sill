#ifndef SILL_CRF_PARAMETER_LEARNER_BUILDER_HPP
#define SILL_CRF_PARAMETER_LEARNER_BUILDER_HPP

#include <boost/program_options.hpp>

#include <sill/learning/crf/crf_parameter_learner.hpp>

namespace sill {

  /**
   * Class for parsing command-line options to create a crf_parameter_learner.
   * @see crf_parameter_learner
   */
  class crf_parameter_learner_builder {

    crf_parameter_learner_parameters cpl_params;

    real_optimizer_builder real_opt_builder;

    std::string learning_objective_string;

  public:

    crf_parameter_learner_builder()
      : learning_objective_string("MLE") { }

    /**
     * Add options to the given Options Description.
     * Once the Options Description is used to parse argv, this struct will
     * hold the specified values.
     */
    void add_options(boost::program_options::options_description& desc,
                     const std::string& desc_prefix = "");

    //! Return the CRF Parameter Learner options specified in this builder.
    const crf_parameter_learner_parameters& get_parameters();

    //! Get the parsed learning objective.
    crf_parameter_learner_parameters::learning_objective_enum
    learning_objective() const;

  }; // class crf_parameter_learner_builder

}  // namespace sill

#endif // SILL_CRF_PARAMETER_LEARNER_BUILDER_HPP

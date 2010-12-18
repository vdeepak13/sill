#ifndef SILL_LEARN_CRF_FACTOR_BUILDER_HPP
#define SILL_LEARN_CRF_FACTOR_BUILDER_HPP

#include <sill/learning/learn_crf_factor.hpp>

namespace sill {

  /**
   * Class for parsing command-line options to run learn_crf_factor.
   */
  class learn_crf_factor_builder {

    table_crf_factor::parameters tcf_params;

    gaussian_crf_factor::parameters gcf_params;

  public:

    learn_crf_factor_builder() { }

    /**
     * Add options to the given Options Description.
     * Once the Options Description is used to parse argv, this struct will
     * hold the specified values.
     */
    void add_options(boost::program_options::options_description& desc);

    //! Return the options for the CRF factor type specified as a template
    //! parameter
    template <typename CRFfactor>
    const typename CRFfactor::parameters& get_parameters();

  }; // class learn_crf_factor_builder

  //! Specialization for table_crf_factor.
  template <>
  const table_crf_factor::parameters& get_parameters<table_crf_factor>();

  //! Specialization for gaussian_crf_factor.
  template <>
  const gaussian_crf_factor::parameters& get_parameters<gaussian_crf_factor>();

}  // namespace sill

#endif // SILL_LEARN_CRF_FACTOR_BUILDER_HPP

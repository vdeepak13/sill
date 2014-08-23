#include <sill/learning/crf/learn_crf_factor_builder.hpp>

namespace sill {

  void learn_crf_factor_builder::add_options
  (boost::program_options::options_description& desc) {
    
  } // add_options

  template <>
  const table_crf_factor::parameters&
  learn_crf_factor_builder::get_parameters<table_crf_factor>() {
    return tcf_params;
  }

  //! Specialization for gaussian_crf_factor.
  template <>
  const gaussian_crf_factor::parameters&
  learn_crf_factor_builder::get_parameters<gaussian_crf_factor>() {
    return gcf_params;
  }

}; // namespace sill

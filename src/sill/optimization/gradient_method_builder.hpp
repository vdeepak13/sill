
#ifndef SILL_GRADIENT_METHOD_BUILDER_HPP
#define SILL_GRADIENT_METHOD_BUILDER_HPP

#include <sill/optimization/gradient_method.hpp>
#include <sill/optimization/line_search_builder.hpp>
#include <sill/optimization/single_opt_step_builder.hpp>

namespace sill {

  /**
   * Class for parsing command-line options which specify
   * gradient_method_parameters.
   */
  class gradient_method_builder {

    gradient_method_parameters gm_params;

    line_search_builder ls_builder;

    single_opt_step_builder sos_builder;

    std::string step_type_string;

    std::string ls_stopping_string;

  public:

    gradient_method_builder()
      : step_type_string("line_search"), ls_stopping_string("ls_exact") { }

    /**
     * Add options to the given Options Description.
     * Once the Options Description is used to parse argv, this struct will
     * hold the specified values.
     */
    void add_options(boost::program_options::options_description& desc,
                     const std::string& desc_prefix = "");

    //! Get the parameters specified by this builder.
    const gradient_method_parameters& get_parameters();

  }; // class gradient_method_builder

} // namespace sill

#endif // #ifndef SILL_GRADIENT_METHOD_BUILDER_HPP

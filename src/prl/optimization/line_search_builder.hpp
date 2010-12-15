
#ifndef PRL_OPTIMIZATION_LINE_SEARCH_BUILDER_HPP
#define PRL_OPTIMIZATION_LINE_SEARCH_BUILDER_HPP

#include <boost/program_options.hpp>

#include <prl/optimization/line_search.hpp>

namespace prl {

  /**
   * Class for parsing command-line options which specify
   * line_search_parameters.
   */
  class line_search_builder {

    line_search_parameters ls_params;

  public:

    line_search_builder() { }

    /**
     * Add options to the given Options Description.
     * Once the Options Description is used to parse argv, this struct will
     * hold the specified values.
     */
    void add_options(boost::program_options::options_description& desc,
                     const std::string& desc_prefix = "");

    //! Return the line_search_parameters held in this builder struct.
    const line_search_parameters& get_parameters() {
      return ls_params;
    }

  }; // class line_search_builder

} // end of namespace: prl

#endif // #ifndef PRL_OPTIMIZATION_LINE_SEARCH_BUILDER_HPP

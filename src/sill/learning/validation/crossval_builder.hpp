#ifndef SILL_CROSSVAL_BUILDER_HPP
#define SILL_CROSSVAL_BUILDER_HPP

#include <boost/program_options.hpp>

#include <sill/parsers/string_functions.hpp>
#include <sill/learning/validation/crossval_parameters.hpp>

namespace sill {

  /**
   * Class for parsing command-line options for cross validation.
   */
  struct crossval_builder {

    //! Indicates whether to do cv or use fixed_vals.
    //! This is not in crossval_parameters, but it is useful for tests.
    bool no_cv;

    //! If do_cv == false, then use these vals instead of doing cv.
    //! This is not in crossval_parameters, but it is useful for tests.
    vec fixed_vals;

    size_t nfolds;

    vec minvals;

    vec maxvals;

    uvec nvals;

    size_t zoom;

    bool real_scale;

    crossval_builder() { }

    /**
     * Add options to the given Options Description.
     * Once the Options Description is used to parse argv, this struct will
     * hold the specified values.
     */
    void add_options(boost::program_options::options_description& desc,
                     const std::string& desc_prefix = "");

    /**
     * Return the CV options specified in this builder.
     * This version:
     *  - sets N automatically to minvals.size() and
     *  - checks to make sure N == maxvals.size() too.
     */
    crossval_parameters get_parameters();

    //! Return the CV options specified in this builder.
    //! @param N   Dimensionality of parameter vector.
    crossval_parameters get_parameters(size_t N);

  }; // class crossval_builder

} // namespace sill

#endif // #ifndef SILL_CROSSVAL_BUILDER_HPP

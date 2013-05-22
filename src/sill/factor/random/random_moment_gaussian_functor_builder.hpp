#ifndef SILL_RANDOM_MOMENT_GAUSSIAN_FUNCTOR_BUILDER_HPP
#define SILL_RANDOM_MOMENT_GAUSSIAN_FUNCTOR_BUILDER_HPP

#include <sill/factor/random/random_factor_functor_builder_i.hpp>
#include <sill/factor/random/random_moment_gaussian_functor.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! \addtogroup factor_random
  //! @{

  /**
   * Helper struct for random_moment_gaussian_functor which allows easy parsing
   * of command-line options via Boost Program Options.
   *
   * Usage: Create your own Options Description desc.
   *        Call this struct's add_options() method with desc.
   *        Parse the command line using the modified options description.
   *        Use this struct's create_functor() method to create a functor with
   *        the parsed options.
   */
  struct random_moment_gaussian_functor_builder
    : random_factor_functor_builder_i<random_moment_gaussian_functor> {

    typedef random_moment_gaussian_functor rff_type;

    /**
     * Add options to the given Options Description.
     * Once the Options Description is used to parse argv, this struct will
     * hold the specified values.
     *
     * @param opt_prefix  Prefix added to command line option names.
     *                    This is useful when using multiple functor instances.
     *                    (default = "")
     */
    void add_options(boost::program_options::options_description& desc,
                     const std::string& opt_prefix = "");

    //! Check options.  Assert false if invalid.
    void check() const;

    //! Get the parsed options.
    const rff_type::parameters& get_parameters() const;

    //! Print the options in this struct.
    void print(std::ostream& out) const;

  private:
    mutable rff_type::parameters params;

  }; // struct random_moment_gaussian_functor_builder

  //! @} group factor_random

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_RANDOM_MOMENT_GAUSSIAN_FUNCTOR_BUILDER_HPP

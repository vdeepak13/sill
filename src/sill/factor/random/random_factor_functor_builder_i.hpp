#ifndef SILL_RANDOM_FACTOR_FUNCTOR_BUILDER_I_HPP
#define SILL_RANDOM_FACTOR_FUNCTOR_BUILDER_I_HPP

#include <boost/program_options.hpp>

namespace sill {

  //! \addtogroup factor_random
  //! @{

  /**
   * Interface for helper structs for random_factor_functor_i and
   * random_crf_factor_functor_i types which allow
   * easy parsing of command-line options via Boost Program Options.
   *
   * NOTE: This same interface is used for both regular and CRF factors.
   *
   * Usage: Create your own Options Description desc.
   *        Call this struct's add_options() method with desc.
   *        Parse the command line using the modified options description.
   *        Use this struct's create_functor() method to create a functor with
   *        the parsed options.
   *
   * @tparam RFF   Random factor functor type
   */
  template <typename RFF>
  struct random_factor_functor_builder_i {

    typedef RFF rff_type;

    /**
     * Add options to the given Options Description.
     * Once the Options Description is used to parse argv, this struct will
     * hold the specified values.
     */
    virtual
    void add_options(boost::program_options::options_description& desc) = 0;

    //! Check options.  Assert false if invalid.
    virtual
    void check() const = 0;

    //! Get the parsed options.
    virtual
    const typename rff_type::parameters&
    get_parameters() const = 0;

  }; // struct random_factor_functor_builder_i

  //! @} group factor_random

} // namespace sill

#endif // SILL_RANDOM_FACTOR_FUNCTOR_BUILDER_I_HPP

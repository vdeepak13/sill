#ifndef SILL_RANDOM_ALTERNATING_FACTOR_FUNCTOR_BUILDER_HPP
#define SILL_RANDOM_ALTERNATING_FACTOR_FUNCTOR_BUILDER_HPP

#include <sill/factor/random/random_alternating_factor_functor.hpp>
#include <sill/factor/random/random_factor_functor_builder_i.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! \addtogroup factor_random
  //! @{

  /**
   * Helper struct for random_alternating_factor_functor which allows
   * easy parsing of command-line options via Boost Program Options.
   *
   * Usage: Create your own Options Description desc.
   *        Call this struct's add_options() method with desc.
   *        Parse the command line using the modified options description.
   *        Use this struct's create_functor() method to create a functor with
   *        the parsed options.
   *
   * @tparam RFFB  base random_factor_functor_i builder type
   */
  template <typename RFFB>
  struct random_alternating_factor_functor_builder
    : random_factor_functor_builder_i<random_alternating_factor_functor
                                      <typename RFFB::rff_type> > {

    typedef typename RFFB::rff_type                         sub_rff_type;
    typedef random_alternating_factor_functor<sub_rff_type> rff_type;
    typedef random_factor_functor_builder_i<rff_type>       base;

    /**
     * Add options to the given Options Description.
     * Once the Options Description is used to parse argv, this struct will
     * hold the specified values.
     */
    void add_options(boost::program_options::options_description& desc) {

      namespace po = boost::program_options;

      po::options_description
        sub_desc("random_alternating_factor_functor options");
      sub_desc.add_options()
        ("alternation_period",
         po::value<size_t>(&(params.alternation_period))->default_value(2),
         "Alternation period (> 0) If 1, then only alternate_rff is used.");
      desc.add(sub_desc);
      default_rffb.add_options(desc, "def_");
      alternate_rffb.add_options(desc, "alt_");

    } // add_options

    //! Check options.  Assert false if invalid.
    void check() const {
      default_rffb.check();
      alternate_rffb.check();
      params.check();
    } // check

    //! Get the parsed options.
    const typename rff_type::parameters&
    get_parameters() const {
      params.default_rff.params = default_rffb.get_parameters();
      params.alternate_rff.params = alternate_rffb.get_parameters();
      return params;
    }

  private:
    RFFB default_rffb;
    RFFB alternate_rffb;

    mutable typename rff_type::parameters params;

  }; // struct random_alternating_factor_functor_builder

  //! @} group factor_random

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_RANDOM_ALTERNATING_FACTOR_FUNCTOR_BUILDER_HPP

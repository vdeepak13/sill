#ifndef SILL_ALTERNATING_CRF_FACTOR_FUNCTOR_BUILDER_HPP
#define SILL_ALTERNATING_CRF_FACTOR_FUNCTOR_BUILDER_HPP

#include <sill/factor/random/alternating_crf_factor_functor.hpp>
#include <sill/factor/random/random_factor_functor_builder_i.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! \addtogroup factor_random
  //! @{

  /**
   * Helper struct for alternating_crf_factor_functor which allows
   * easy parsing of command-line options via Boost Program Options.
   *
   * Usage: Create your own Options Description desc.
   *        Call this struct's add_options() method with desc.
   *        Parse the command line using the modified options description.
   *        Use this struct's create_functor() method to create a functor with
   *        the parsed options.
   *
   * @tparam RFFB  base random_crf_factor_functor_i builder type
   */
  template <typename RFFB>
  struct alternating_crf_factor_functor_builder
    : random_factor_functor_builder_i<alternating_crf_factor_functor
                                      <typename RFFB::rff_type> > {

    typedef typename RFFB::rff_type                        sub_rff_type;
    typedef alternating_crf_factor_functor<sub_rff_type>   rff_type;
    typedef random_factor_functor_builder_i<rff_type>  base;

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
                     const std::string& opt_prefix = "") {

      namespace po = boost::program_options;

      po::options_description
        sub_desc("alternating_crf_factor_functor "
                 +(opt_prefix == "" ? std::string("") : "(" + opt_prefix + ") ")
                 + "options");
      sub_desc.add_options()
        ((opt_prefix + "alternation_period").c_str(),
         po::value<size_t>(&(params.alternation_period))->default_value(2),
         "Alternation period (> 0) If 1, then only alternate_rff is used.");
      desc.add(sub_desc);
      default_rffb.add_options(desc, opt_prefix + "def_");
      alternate_rffb.add_options(desc, opt_prefix + "alt_");

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

    //! Print the options in this struct.
    void print(std::ostream& out) const {
      out << "alternation_period: " << params.alternation_period << "\n"
          << "default_rffb:\n"
          << default_rffb << "\n"
          << "alternate_rffb:\n"
          << alternate_rffb << "\n";
    }

  private:
    RFFB default_rffb;
    RFFB alternate_rffb;

    mutable typename rff_type::parameters params;

  }; // struct alternating_crf_factor_functor_builder

  //! @} group crf_factor_random

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_ALTERNATING_CRF_FACTOR_FUNCTOR_BUILDER_HPP

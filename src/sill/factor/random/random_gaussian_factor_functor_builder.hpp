#ifndef SILL_RANDOM_GAUSSIAN_FACTOR_FUNCTOR_BUILDER_HPP
#define SILL_RANDOM_GAUSSIAN_FACTOR_FUNCTOR_BUILDER_HPP

#include <sill/factor/random/random_factor_functor_builder_i.hpp>
#include <sill/factor/random/random_gaussian_factor_functor.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! \addtogroup factor_random
  //! @{

  /**
   * Helper struct for random_gaussian_factor_functor which allows easy parsing
   * of command-line options via Boost Program Options.
   *
   * Usage: Create your own Options Description desc.
   *        Call this struct's add_options() method with desc.
   *        Parse the command line using the modified options description.
   *        Use this struct's create_functor() method to create a functor with
   *        the parsed options.
   *
   * @tparam F   Gaussian factor type.
   */
  template <typename F>
  struct random_gaussian_factor_functor_builder
    : random_factor_functor_builder_i<random_gaussian_factor_functor<F> > {

    typedef random_gaussian_factor_functor<F> rff_type;

    /**
     * Add options to the given Options Description.
     * Once the Options Description is used to parse argv, this struct will
     * hold the specified values.
     */
    void add_options(boost::program_options::options_description& desc) {
      add_options(desc, "");
    }

    /**
     * Add options to the given Options Description.
     * Once the Options Description is used to parse argv, this struct will
     * hold the specified values.
     *
     * @param opt_prefix  Prefix added to command line option names.
     *                    This is useful when using multiple functor instances.
     */
    void add_options(boost::program_options::options_description& desc,
                     const std::string& opt_prefix) {

      namespace po = boost::program_options;

      po::options_description
        sub_desc("random_gaussian_factor_functor "
                 + (opt_prefix=="" ? std::string("") : "(" + opt_prefix + ") ")
                 + "options");
      sub_desc.add_options()
        ((opt_prefix + "b").c_str(),
         po::value<double>(&(params.b))->default_value(1),
         "Each element of the mean is chosen from Uniform[-b, b].  (b >= 0)")
        ((opt_prefix + "variance").c_str(),
         po::value<double>(&(params.variance))->default_value(1),
         "Set variances of each variable to this value.  (variance > 0)")
        ((opt_prefix + "correlation").c_str(),
         po::value<double>(&(params.correlation))->default_value(.3),
         "Set covariance of each pair of variables according to this correlation coefficient.  (fabs(correlation) <= 1)")
        ((opt_prefix + "c").c_str(),
         po::value<double>(&(params.c))->default_value(1),
         "Each element of the coefficient matrix C is chosen from c_shift + Uniform[-c, c], where C shifts the mean when conditioning on X=x.  (c >= 0)")
        ((opt_prefix + "c_shift").c_str(),
         po::value<double>(&(params.c_shift))->default_value(0),
         "(See option c)");
      desc.add(sub_desc);
    } // add_options

    //! Check options.  Assert false if invalid.
    void check() const {
      params.check();
    }

    //! Get the parsed options.
    const typename rff_type::parameters& get_parameters() const {
      return params;
    }

  private:
    mutable typename rff_type::parameters params;

  }; // struct random_gaussian_factor_functor_builder

  //! @} group factor_random

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_RANDOM_GAUSSIAN_FACTOR_FUNCTOR_BUILDER_HPP

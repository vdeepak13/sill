#ifndef SILL_MOMENT_GAUSSIAN_FN_BUILDER_HPP
#define SILL_MOMENT_GAUSSIAN_FN_BUILDER_HPP

#include <sill/factor/random/alternating_generator.hpp>
#include <sill/factor/random/moment_gaussian_generator.hpp>
#include <sill/factor/random/functional.hpp>

#include <boost/program_options.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A class that is able to parse the parameters of moment_gaussian
   * generator from Boost Program Options and return factor functors
   * that generate random moment_gaussian factors according to these parameters.
   * 
   * To use this class, first call add_options to register options
   * within the given description. After argv is parsed, use can invoke
   * marginal_fn(), and conditional_fn() to retrieve the functors
   * corresponding to the specified parameters.
   * 
   * \ingroup factor_random
   */
  class moment_gaussian_fn_builder {
  public:
    moment_gaussian_fn_builder() { }

    /**
     * Add options to the given Options Description.
     *
     * @param opt_prefix Prefix added to command line option names.
     *                   This is useful when using multiple functor instances.
     */
    void add_options(boost::program_options::options_description& desc,
                     const std::string& opt_prefix = "") {
      namespace po = boost::program_options;
      po::options_description
        sub_desc("moment_gaussian_generator "
             + (opt_prefix.empty() ? std::string() : "(" + opt_prefix + ") ")
             + "options");
      sub_desc.add_options()
        ((opt_prefix + "period").c_str(),
         po::value<size_t>(&period)->default_value(0),
         "Alternation period. If 0, only the default is used.");
      add_options(sub_desc, opt_prefix, def);
      add_options(sub_desc, opt_prefix + "alt_", alt);
      desc.add(sub_desc);
    }
    
    /**
     * Returns a functor that generates random marginals according to the
     * parameters specified by the parsed Boost program options.
     * \param rng The random number generator used to generate the marginals.
     */
    template <typename RandomNumberGenerator>
    marginal_moment_gaussian_fn
    marginal_fn(RandomNumberGenerator& rng) const {
      if (period == 0) {
        // regular generator
        moment_gaussian_generator gen(def);
        return sill::marginal_fn(gen, rng);
      } else {
        // alternating generator
        moment_gaussian_generator gen1(def);
        moment_gaussian_generator gen2(alt);
        return sill::marginal_fn(make_alternating_generator(gen1, gen2, period), rng);
      }
    }

    /**
     * Returns a functor that generates random conditionals according to the
     * parameters specified by the parsed Boost program options.
     * \param rng 
     */
    template <typename RandomNumberGenerator>
    conditional_moment_gaussian_fn
    conditional_fn(RandomNumberGenerator& rng) const {
      if (period == 0) {
        // regular generator
        moment_gaussian_generator gen(def);
        return sill::conditional_fn(gen, rng);
      } else {
        // alternating generator
        moment_gaussian_generator gen1(def);
        moment_gaussian_generator gen2(alt);
        return sill::conditional_fn(make_alternating_generator(gen1, gen2, period), rng);
      }
    }

  private:
    //! Parameters that can be specified on the command line
    typedef moment_gaussian_generator::param_type param_type;

    void add_options(boost::program_options::options_description& desc,
                     const std::string& opt_prefix,
                     param_type& params) {
      namespace po = boost::program_options;
      desc.add_options()
        ((opt_prefix + "mean_lower").c_str(),
         po::value<double>(&(params.mean_lower))->default_value(-1.0),
         "Each element of the mean is chosen from Uniform[mean_lower,mean_upper].")
        ((opt_prefix + "mean_upper").c_str(),
         po::value<double>(&(params.mean_upper))->default_value(1.0),
         "Each element of the mean is chosen from Uniform[mean_lower,mean_upper].")
        ((opt_prefix + "variance").c_str(),
         po::value<double>(&(params.variance))->default_value(1.0),
         "Set the variance of each variable to this value. (variance > 0)")
        ((opt_prefix + "correlation").c_str(),
         po::value<double>(&(params.correlation))->default_value(.3),
         "Set the correlation of each pair of variables. (-1 < correlation < 1)")
        ((opt_prefix + "coeff_lower").c_str(),
         po::value<double>(&(params.coeff_lower))->default_value(-1.0),
         "Each element of the coefficient matrix C is chosen from Uniform[coef_lower,coeff_upper].")
        ((opt_prefix + "coeff_upper").c_str(),
         po::value<double>(&(params.coeff_upper))->default_value(1.0),
         "Each element of the coefficient matrix C is chosen from Uniform[coef_lower,coeff_upper].");        
      desc.add(desc);
    }

  private:
    size_t period;
    param_type def;
    param_type alt;

    friend std::ostream&
    operator<<(std::ostream& out, const moment_gaussian_fn_builder& b) {
      out << b.period << " ";
      if (b.period == 0) {
        out << "(" << b.def << ")";
      } else {
        out << "def(" << b.def << ") alt(" << b.alt << ")";
      }
      return out;
    }

  }; // class moment_gaussian_fn_builder

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

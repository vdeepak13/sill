// this stuff may not be needed if we can rely on automatic conversions
// from moment_gaussian to canonical_gaussian (not sure if this is a good idea)

#ifndef SILL_CANONICAL_GAUSSIAN_FN_BUILDER_HPP
#define SILL_CANONICAL_GAUSSIAN_FN_BUILDER_HPP

#include <sill/factor/random/moment_gaussian_fn_builder.hpp>
#include <sill/factor/canonical_gaussian.hpp>

namespace sill {

  /**
   * A class that is able to parse parameters of moment_gaussian
   * generator from Boost Program Options and return CRF factor
   * functors that generate random Gaussian CRF factors according
   * to these parameters.
   *
   * The class simply delegates to moment_gaussian_fn_builder and
   * converts the returned moment_gaussian objects to canonical_gaussian.
   *
   * \ingroup factor_random
   */
  class canonical_gaussian_fn_builder {
  public:
    canonical_gaussian_fn_builder() { }

    /**
     * Add options to the given Options Description.
     */
    void add_options(boost::program_options::options_description& desc,
                     const std::string& opt_prefix = "") {
      base_builder.add_options(desc, opt_prefix);
    }

    /**
     * Returns a functor that generates random canonical Gaussian factors
     * according to the parameters specified by the parsed Boost program options.
     * \param rng The random number generator used to generate the factors.
     */
    template <typename RandomNumberGenerator>
    marginal_canonical_gaussian_fn
    factor_fn(RandomNumberGenerator& rng) const {
      return marginal_wrapper(base_builder.marginal_fn(rng));
    }

    template <typename RandomNumberGenerator>
    conditional_canonical_gaussian_fn
    factor_fn(RandomNumberGenerator& rng) const {
      return conditional_wrapper(base_builder.conditional_fn(rng));
    }
    
  private:
    moment_gaussian_fn_builder base_builder;

    struct marginal_wrapper {
      marginal_moment_gaussian_fn base_fn;

      wrapper(const marginal_moment_gaussian_fn& base_fn)
        : base_fn(base_fn) { }

      canonical_gaussian operator()(const vector_domain& args) {
        return canonical_gaussian(base_fn(args));
      }
    }; // struct marginal_wrapper

    struct conditional_wrapper {
      conditional_moment_gaussian_fn base_fn;

      wrapper(const conditional_moment_gaussian_fn& base_fn)
        : base_fn(base_fn) { }

      canonical_gaussian operator()(const vector_domain& head,
                                    const vector_domain& tail) {
        return canonical_gaussian(base_fn(head, tail));
      }
    }; // struct conditional_wrapper
    
  }; // class canonical_gaussian_fn_builder

} // namespace sill

#endif

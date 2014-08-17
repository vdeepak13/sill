#ifndef SILL_GAUSSIAN_CRF_FACTOR_FN_BUILDER_HPP
#define SILL_GAUSSIAN_CRF_FACTOR_FN_BUILDER_HPP

#include <sill/factor/random/moment_gaussian_fn_builder.hpp>
#include <sill/factor/gaussian_crf_factor.hpp>

namespace sill {

  /**
   * A class that is able to parse parameters of moment_gaussian
   * generator from Boost Program Options and return CRF factor
   * functors that generate random Gaussian CRF factors according
   * to these parameters.
   *
   * The class simply delegates to moment_gaussian_fn_builder and
   * converts the returned moment_gaussian objects to gaussian_crf_factor.
   *
   * \ingroup factor_random
   */
  class gaussian_crf_factor_fn_builder {
  public:
    gaussian_crf_factor_fn_builder() { }

    /**
     * Add options to the given Options Description.
     */
    void add_options(boost::program_options::options_description& desc,
                     const std::string& opt_prefix = "") {
      base_builder.add_options(desc, opt_prefix);
    }

    /**
     * Returns a functor that generates random gaussian CRF factors
     * according to the parameters specified by the parsed Boost program options.
     * \param rng The random number generator used to generate the factors.
     */
    template <typename RandomNumberGenerator>
    gaussian_crf_factor_fn
    factor_fn(RandomNumberGenerator& rng) const {
      return wrapper(base_builder.conditional_fn(rng));
    }
    
  private:
    moment_gaussian_fn_builder base_builder;

    struct wrapper {
      conditional_moment_gaussian_fn base_fn;

      wrapper(const conditional_moment_gaussian_fn& base_fn)
        : base_fn(base_fn) { }

      gaussian_crf_factor operator()(const vector_domain& head,
                                     const vector_domain& tail) {
        return gaussian_crf_factor(base_fn(head, tail), head, tail);
      }
    }; // struct wrapper
    
    friend std::ostream&
    operator<<(std::ostream& out, const gaussian_crf_factor_fn_builder& b) {
      out << b.base_builder;
      return out;
    }
    
  }; // class gaussian_crf_factor_fn_builder

} // namespace sill

#endif

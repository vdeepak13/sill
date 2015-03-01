#ifndef SILL_TABLE_CRF_FACTOR_FN_BUILDER_HPP
#define SILL_TABLE_CRF_FACTOR_FN_BUILDER_HPP

#include <sill/factor/random/table_factor_fn_builder.hpp>
#include <sill/factor/crf/table_crf_factor.hpp>

namespace sill {

  /**
   * A class that is able to parse parameters of table factor
   * generators from Boost Program Options and return CRF factor
   * functors that generate random CRF table factors according
   * to these parameters.
   *
   * The class simply delegates to table_factor_fn_builder and
   * converts the returned table_factor objects to  table_crf_factor.
   *
   * \ingroup factor_random
   */
  class table_crf_factor_fn_builder {
  public:
    table_crf_factor_fn_builder() { }

    /**
     * Add options to the given Options Description.
     */
    void add_options(boost::program_options::options_description& desc,
                     const std::string& opt_prefix = "") {
      base_builder.add_options(desc, opt_prefix);
    }

    /**
     * Returns a functor that generates random table CRF factors
     * according to the parameters specified by the parsed Boost program options.
     * \param rng The random number generator used to generate the factors.
     */
    template <typename RandomNumberGenerator>
    table_crf_factor_fn
    factor_fn(RandomNumberGenerator& rng) const {
      return wrapper(base_builder.conditional_fn(rng));
    }

    const std::string& get_kind() const {
      return base_builder.get_kind();
    }
    
  private:
    table_factor_fn_builder base_builder;

    struct wrapper {
      conditional_table_factor_fn base_fn;

      wrapper(const conditional_table_factor_fn& base_fn)
        : base_fn(base_fn) { }

      table_crf_factor operator()(const finite_domain& head,
                                  const finite_domain& tail) {
        table_crf_factor f(base_fn(head, tail), head, false);
        f.convert_to_log_space();
        return f;
      }
    }; // struct wrapper
    
    friend std::ostream&
    operator<<(std::ostream& out, const table_crf_factor_fn_builder& b) {
      out << b.base_builder;
      return out;
    }
  }; // class table_crf_factor_fn_builder

} // namespace sill

#endif

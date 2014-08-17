#ifndef SILL_TABLE_FACTOR_FN_BUILDER_HPP
#define SILL_TABLE_FACTOR_FN_BUILDER_HPP

#include <sill/factor/random/alternating_generator.hpp>
#include <sill/factor/random/associative_factor_generator.hpp>
#include <sill/factor/random/discrete_factor_generator.hpp>
#include <sill/factor/random/ising_factor_generator.hpp>
#include <sill/factor/random/uniform_factor_generator.hpp>
#include <sill/factor/random/functional.hpp>

#include <boost/program_options.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A class that is able to parse the parameters of various table factor
   * generators from Boost Program Options and return factor functors
   * that generate random table factors according to these paramters.
   *
   * To use this class, first call add_options to register options
   * within the given description. After argv is parsed, use can invoke
   * marginal_fn(), and conditional_fn() to retrieve the functors
   * corresponding to the specified parameters.
   * 
   * \ingroup factor_random
   */
  class table_factor_fn_builder {
  public:
    table_factor_fn_builder() { }

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
        sub_desc("table_factor generator "
             + (opt_prefix.empty() ? std::string() : "(" + opt_prefix + ") ")
             + "options");
      sub_desc.add_options()
        ((opt_prefix + "kind").c_str(),
         po::value<std::string>(&kind)->default_value("filled"),
         "Kind of factor: filled/ising/associative")
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
    marginal_table_factor_fn
    marginal_fn(RandomNumberGenerator& rng) const {
      if (period == 0) {
        // regular generators
        if (kind == "associative") {
          associative_factor_generator gen(def.lower, def.upper);
          return sill::marginal_fn(gen, rng);
        }
        if (kind == "discrete") {
          discrete_factor_generator gen(def.alpha);
          return sill::marginal_fn(gen, rng);
        }
        if (kind == "ising") {
          ising_factor_generator gen(def.lower, def.upper);
          return sill::marginal_fn(gen, rng);
        }
        if (kind == "uniform") {
          uniform_factor_generator gen(def.lower, def.upper);
          return sill::marginal_fn(gen, rng);
        }
      } else {
        // alternating generators
        if (kind == "associative") {
          associative_factor_generator gen1(def.lower, def.upper);
          associative_factor_generator gen2(alt.lower, alt.upper);
          return sill::marginal_fn(make_alternating_generator(gen1, gen2, period), rng);
        }
        if (kind == "discrete") {
          discrete_factor_generator gen1(def.alpha);
          discrete_factor_generator gen2(alt.alpha);
          return sill::marginal_fn(make_alternating_generator(gen1, gen2, period), rng);
        }
        if (kind == "ising") {
          ising_factor_generator gen1(def.lower, def.upper);
          ising_factor_generator gen2(alt.lower, alt.upper);
          return sill::marginal_fn(make_alternating_generator(gen1, gen2, period), rng);
        }
        if (kind == "uniform") {
          uniform_factor_generator gen1(def.lower, def.upper);
          uniform_factor_generator gen2(alt.lower, alt.upper);
          return sill::marginal_fn(make_alternating_generator(gen1, gen2, period), rng);
        }
      }
      throw std::invalid_argument("Invalid table factor kind: "+kind);
    }

    /**
     * Returns a functor that generates random conditionals according to the
     * parameters specified by the parsed Boost program options.
     * \param rng 
     */
    template <typename RandomNumberGenerator>
    conditional_table_factor_fn
    conditional_fn(RandomNumberGenerator& rng) const {
      if (period == 0) {
        // regular generators
        if (kind == "discrete") {
          discrete_factor_generator gen(def.alpha);
          return sill::conditional_fn(gen, rng);
        }
        if (kind == "uniform") {
          uniform_factor_generator gen(def.lower, def.upper);
          return sill::conditional_fn(gen, rng);
        }
      } else {
        // alternating generators
        if (kind == "discrete") {
          discrete_factor_generator gen1(def.alpha);
          discrete_factor_generator gen2(alt.alpha);
          return sill::conditional_fn(make_alternating_generator(gen1, gen2, period), rng);
        }
        if (kind == "uniform") {
          uniform_factor_generator gen1(def.lower, def.upper);
          uniform_factor_generator gen2(alt.lower, alt.upper);
          return sill::conditional_fn(make_alternating_generator(gen1, gen2, period), rng);
        }
      }
      throw std::invalid_argument("Invalid table factor kind: " + kind);
    }

    const std::string& get_kind() const {
      return kind;
    }

  private:
    /**
     * The union of all table_factor generator parameters. For simplicity,
     * we only use the two-parameter version of ising_factor_generator.
     */
    struct param_type {
      double lower;
      double upper;
      double alpha;

      friend std::ostream& operator<<(std::ostream& out, const param_type& p) {
        out << p.lower << " " << p.upper << " " << p.alpha;
        return out;
      }
    };

    void add_options(boost::program_options::options_description& desc,
                     const std::string& opt_prefix,
                     param_type& params) {
      namespace po = boost::program_options;
      desc.add_options()
        ((opt_prefix + "lower").c_str(),
         po::value<double>(&params.lower)->default_value(0.0),
         "Lower bound for factor parameters (in log space)")
        ((opt_prefix + "upper").c_str(),
         po::value<double>(&params.upper)->default_value(1.0),
         "Upper bound for factor parameters (in log space)")
        ((opt_prefix + "alpha").c_str(),
         po::value<double>(&params.alpha)->default_value(1.0),
         "The concentration parameter of the Dirichlet distribution");
    }

    std::string kind;
    size_t period;
    param_type def;
    param_type alt;

    friend std::ostream&
    operator<<(std::ostream& out, const table_factor_fn_builder& b) {
      out << b.kind << " " << b.period << " ";
      if (b.period == 0) {
        out << "(" << b.def << ")";
      } else {
        out << "def(" << b.def << ") alt(" << b.alt << ")";
      }
      return out;
    }

  }; // class table_factor_fn_builder

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

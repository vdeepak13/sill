#ifndef SILL_ASSOCIATIVE_FACTOR_GENERATOR_HPP
#define SILL_ASSOCIATIVE_FACTOR_GENERATOR_HPP

#include <sill/factor/table_factor.hpp>

#include <boost/random/uniform_real.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Functor for generating random associative factors.
   * The variables in associative factors must all have the same cardinality.
   * Associative factors assign value exp(x_k) drawn from Uniform[lower, upper]
   * to each tuple of assignments (k, ..., k) and 1.0 everywhere else.
   * 
   * \see RandomMarginalFactorGenerator
   * \ingroup factor_random
   */
  class associative_factor_generator {
  public:
    // RandomMarginalFactorGenerator typedefs
    typedef finite_domain domain_type;
    typedef table_factor  result_type;

    struct param_type {
      double lower;
      double upper;

      param_type()
        : lower(0.0), upper(1.0) { }

      param_type(double lower, double upper)
        : lower(lower), upper(upper) {
        check();
      }

      void check() const {
        assert(lower <= upper);
      }

      friend std::ostream& operator<<(std::ostream& out, const param_type& p) {
        out << p.lower << " " << p.upper;
        return out;
      }
    }; // struct param_type

    //! Constructs generator of associative factors with the given limits
    explicit associative_factor_generator(double lower = 0.0, double upper = 1.0)
      : params(lower, upper) { }
    
    //! Constructs generator with the given parameters
    explicit associative_factor_generator(const param_type& params)
      : params(params) { params.check(); }

    //! Generate a marginal distribution p(args) using the stored parameters.
    template <typename RandomNumberGenerator>
    table_factor operator()(const finite_domain& args,
                            RandomNumberGenerator& rng) {
      if (args.empty()) {
        return table_factor(1.0);
      }
      boost::uniform_real<> unif(params.lower, params.upper);
      std::vector<double> values((*args.begin())->size());
      foreach(double& x, values) { x = unif(rng); }
      return make_associative_factor(args, values);
    }

    //! Returns the parameter set associated with this generator
    const param_type& param() const {
      return params;
    }

    //! Sets the parameter set associated with this generator
    void param(const param_type& params) {
      params.check();
      this->params = params;
    }

  private:
    param_type params;

  }; // class associative_factor_generator

  //! Prints the parameters of the generator to an output stream.
  //! \relates associative_factor_generator
  inline std::ostream&
  operator<<(std::ostream& out, const associative_factor_generator& gen) {
    out << gen.param();
    return out;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

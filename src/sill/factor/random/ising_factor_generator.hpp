#ifndef SILL_ISING_FACTOR_GENERATOR_HPP
#define SILL_ISING_FACTOR_GENERATOR_HPP

#include <sill/factor/table_factor.hpp>

#include <stdexcept>

#include <boost/random/uniform_real.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Functor for generating random Ising factors. Ising factors can have at
   * most two variables, where each variable is itself binary (takes on two
   * values).  Unary Ising factors assign value exp(x) to value 1 and
   * exp(-x) to value 0 for x drawn from Uniform[lower, upper].
   * Binary Ising factors assign value exp(x) to the diagonal and exp(-x)
   * to the off-diagonal, where x is drawn once from Uniform[lower, upper].
   *
   * \see RandomMarginalFactorGenerator
   * \ingroup factor_random
   */
  class ising_factor_generator {
  public:
    // RandomMarginalFactorGenerator typedefs
    typedef finite_domain domain_type;
    typedef table_factor  result_type;

    struct param_type {
      double node_lower;
      double node_upper;
      double edge_lower;
      double edge_upper;

      param_type()
        : node_lower(0.0), node_upper(1.0),
          edge_lower(0.0), edge_upper(1.0) { }

      param_type(double node_lower, double node_upper,
                 double edge_lower, double edge_upper)
        : node_lower(node_lower), node_upper(node_upper),
          edge_lower(edge_lower), edge_upper(edge_upper) {
        check();
      }

      void check() const {
        assert(node_lower <= node_upper);
        assert(edge_lower <= edge_upper);
      }

      friend std::ostream& operator<<(std::ostream& out, const param_type& p) {
        out << p.node_lower << " " << p.node_upper << " "
            << p.edge_lower << " " << p.edge_upper;
        return out;
      }
    }; // struct param_type

    //! Constructs a generator of Ising factors with equal node and edge limits
    explicit ising_factor_generator(double lower = 0.0, double upper = 1.0)
      : params(lower, upper, lower, upper) { }

    //! Constructs a generator of Ising factors with specified limits
    ising_factor_generator(double node_lower, double node_upper,
                           double edge_lower, double edge_upper)
      : params(node_lower, node_upper, edge_lower, edge_upper) { }

    //! Constructs generator with the given parameters
    explicit ising_factor_generator(const param_type& params)
      : params(params) { params.check(); }

    //! Generate a marginal distribution p(args) using the stored parameters.
    template <typename RandomNumberGenerator>
    table_factor operator()(const finite_domain& args,
                            RandomNumberGenerator& rng) {
      switch (args.size()) {
      case 0: {
        return table_factor(1.0);
      }
      case 1: {
        boost::uniform_real<> unif(params.node_lower, params.node_upper);
        return make_ising_factor(*args.begin(), unif(rng));
      }
      case 2: {
        boost::uniform_real<> unif(params.edge_lower, params.edge_upper);
        return make_ising_factor(*args.begin(), *++args.begin(), unif(rng));
      }
      default: {
        throw std::invalid_argument("ising_factor_generator only supports "
                                    "factors with <=2 variables");
      }
      }
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

  }; // class ising_factor_generator

  //! Prints the parameters of the generator to an output stream.
  //! \relates ising_factor_generator
  inline std::ostream&
  operator<<(std::ostream& out, const ising_factor_generator& gen) {
    out << gen.param();
    return out;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

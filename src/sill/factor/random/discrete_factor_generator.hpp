#ifndef SILL_DISCRETE_FACTOR_GENERATOR_HPP
#define SILL_DISCRETE_FACTOR_GENERATOR_HPP

#include <sill/factor/table_factor.hpp>
#include <sill/math/random.hpp>

#include <boost/random/uniform_real.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Functor for generating table factors that are drawn from a
   * Dirichlet distribution.
   *
   * Marginal factor is simply drawn from Dirichlet(k, alpha),
   * where k is the total number of assignments to the variables.
   * Conditional factors are constructed as follows:
   * For each assignment to the tail variables, the factor over the
   * remaining variables is drawn from Dirichlet(k, alpha),
   * where k is the number of assignments to the head variables.
   *
   * \see RandomFactorGenerator
   * \ingroup factor_random
   */
  class discrete_factor_generator {
  public:
    // RandomFactorGenerator typedefs
    typedef finite_domain domain_type;
    typedef table_factor  result_type;

    struct param_type {
      double alpha;

      param_type()
        : alpha(1.0) { }

      param_type(double alpha)
        : alpha(alpha) {
        check();
      }

      void check() const {
        assert(alpha > 0.0);
      }

      friend std::ostream& operator<<(std::ostream& out, const param_type& p) {
        out << p.alpha;
        return out;
      }
    }; // struct param_type

    //! Constructs a generator of Dirichlet-distributed factors
    explicit discrete_factor_generator(double alpha = 1.0)
      : params(alpha) { }

    //! Constructs generator with the given parameters
    explicit discrete_factor_generator(const param_type& params)
      : params(params) { params.check(); }

    //! Generate a marginal distribution p(args) using the stored parameters.
    template <typename RandomNumberGenerator>
    table_factor operator()(const finite_domain& args,
                            RandomNumberGenerator& rng) {
      size_t k = num_assignments(args);
      dirichlet_distribution<> dirichlet(k, params.alpha);
      return make_dense_table_factor(make_vector(args), dirichlet(rng));
    }

    //! Generates a conditional distribution p(head | tail) using the stored
    //! parameters.
    template <typename RandomNumberGenerator>
    table_factor operator()(const finite_domain& head,
                            const finite_domain& tail,
                            RandomNumberGenerator& rng) {
      // things go horribly wrong if this is not true
      assert(set_disjoint(head, tail));
      table_factor f(sill::concat(make_vector(head), make_vector(tail)));
      size_t k_head = num_assignments(head);
      size_t k_tail = num_assignments(tail);
      dirichlet_distribution<> dirichlet(k_head, params.alpha);
      table_factor::table_type::iterator it = f.table().begin();
      for (size_t i = 0; i < k_tail; ++i) {
        vec vals = dirichlet(rng);
        it = std::copy(vals.begin(), vals.end(), it);
      }
      return f;
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

  }; // class discrete_factor_generator

  //! Prints the parameters of the generator to an output stream.
  //! \relates discrete_factor_generator
  inline std::ostream&
  operator<<(std::ostream& out, const discrete_factor_generator& gen) {
    out << gen.param();
    return out;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

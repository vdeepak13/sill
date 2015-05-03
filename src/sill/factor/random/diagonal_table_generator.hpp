#ifndef SILL_DIAGONAL_TABLE_GENERATOR_HPP
#define SILL_DIAGONAL_TABLE_GENERATOR_HPP

#include <sill/base/finite_variable.hpp>
#include <sill/datastructure/finite_index.hpp>

#include <iostream>
#include <random>
#include <stdexcept>

namespace sill {

  /**
   * A class for generating diagonal table-like factor, including associative
   * factors and ising factor. The variables passed to operator() must all have
   * the same  cardinality. In each call to operator(), the generator then
   * assigns a value x drawn from Uniform[lower, upper] to all diagonal parameters,
   * i.e., the parameters corresponding to each tuple of assignments (k, ..., k).
   * The off-diagonal elements are assigned the probability value 1.0.
   * 
   * \tparam F the generated factor type
   *
   * \see RandomMarginalFactorGenerator
   * \ingroup factor_random
   */
  template <typename F>
  class diagonal_table_generator {
  public:
    // The real type of the factor
    typedef typename F::real_type real_type;

    // RandomMarginalFactorGenerator typedefs
    typedef typename F::domain_type domain_type;
    typedef F result_type;

    struct param_type {
      real_type lower;
      real_type upper;

      param_type(real_type lower = real_type(0),
                 real_type upper = real_type(1))
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
    explicit diagonal_table_generator(real_type lower = real_type(0),
                                      real_type upper = real_type(1))
      : param_(lower, upper) { }
    
    //! Constructs generator with the given parameters
    explicit diagonal_table_generator(const param_type& param)
      : param_(param) { param_.check(); }

    //! Generate a marginal distribution p(args) using the stored parameters.
    template <typename RandomNumberGenerator>
    F operator()(const domain_type& args, RandomNumberGenerator& rng) const {
      F f(args, typename F::result_type(1));
      if (!args.empty()) {
        real_type x =
          std::uniform_real_distribution<real_type>(param_.lower, param_.upper)(rng);
        size_t size = args[0]->size();
        for (finite_variable* v : args) {
          if (v->size() != size) {
            throw std::invalid_argument(
              "diagonal_table_generator: all arguments must have the same size"
            );
          }
        }
        finite_index index;
        for (size_t k = 0; k < size; ++k) {
          index.assign(args.size(), k);
          f.param(index) = x;
        }
      }
      return f;
    }

    //! Returns the parameter set associated with this generator
    const param_type& param() const {
      return param_;
    }

    //! Sets the parameter set associated with this generator
    void param(const param_type& param) {
      param.check();
      param_ = param;
    }

  private:
    param_type param_;

  }; // class diagonal_table_generator

  /**
   * Prints the parameters of the generator to an output stream.
   * \relates diagonal_table_generator
   */
  template <typename F>
  std::ostream&
  operator<<(std::ostream& out, const diagonal_table_generator<F>& gen) {
    out << gen.param();
    return out;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

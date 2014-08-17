#ifndef SILL_MIXTURE_GENERATOR_HPP
#define SILL_MIXTURE_GENERATOR_HPP

#include <sill/factor/mixture.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Functor for generating random mixture factors.
   * \tparam Generator
   *         Class for generating the mixture components. Must satisfy the
   *         RandomFactorGenerator concept.
   *
   * \see RandomFactorGenerator
   * \ingroup factor_random
   */
  template <typename Generator>
  struct mixture_generator {
  public:
    // RandomFactorGenerator typedefs
    typedef typename Generator::domain_type domain_type;
    typedef typename Generator::result_type elem_type;
    typedef mixture<elem_type>              result_type;

    // this is used only to interface with the caller
    struct param_type {
      size_t k;
      typename Generator::param_type base_params;

      explicit param_type(size_t k = 3)
        : k(k) { }

      param_type(size_t k, const typename Generator::param_type& base_params)
        : k(k), base_params(base_params) { }

      friend std::ostream& operator<<(std::ostream& out, const param_type& p) {
        out << p.k << " " << p.base_params;
        return out;
      }
    };

    //! Constructs a generator with k components and default params for Generator
    explicit mixture_generator(size_t k = 3)
      : k(k) { assert(k > 0); }

    //! Constructs a generator with k components and given Generator.
    mixture_generator(size_t k, const Generator& gen)
      : k(k), gen(gen) { assert(k > 0); }

    //! Constructs a generator with k components and Generator's parameter set.
    mixture_generator(size_t k, const typename Generator::param_type& params)
      : k(k), gen(params) { assert(k > 0); }

    //! Constructs a mixture generator with given parameters
    explicit mixture_generator(const param_type& params)
      : k(params.k), gen(params.base_params) { assert(k > 0); }
    
    //! Generate a marginal distribution p(args) using the stored parameters
    template <typename RandomNumberGenerator>
    mixture<elem_type> operator()(const domain_type& args,
                                  RandomNumberGenerator& rng) {
      mixture<elem_type> mix(k, args);
      for (size_t i = 0; i < k; ++i) {
        mix[i] = gen(args, rng);
      }
      return mix;
    }

    //! Generate a conditional distribution p(head | tail) using the stored
    //! parameters.
    template <typename RandomNumberGenerator>
    mixture<elem_type> operator()(const domain_type& head,
                                  const domain_type& tail,
                                  RandomNumberGenerator& rng) {
      mixture<elem_type> mix(k, set_union(head, tail));
      for (size_t i = 0; i < k; ++i) {
        mix[i] = gen(head, tail, rng);
      }
      return mix;
    }

    //! Returns the parameter set associated with this generator
    param_type param() const {
      return param_type(k, gen.param());
    }

    //! Sets the parameter set associated with this generator
    void param(const param_type& params) {
      assert(params.k > 0);
      this->k = params.k;
      gen.param(params.base_params);
    }

  private:
    size_t k;      // the number of components
    Generator gen; // the component generator

  }; // class mixture_generator

  //! Prints the parameters of this generator to an output stream
  //! \relates mixture_generator
  template <typename Generator>
  inline std::ostream&
  operator<<(std::ostream& out, const mixture_generator<Generator>& gen) {
    out << gen.param();
    return out;
  }
  
} // namespace sill

#include <sill/macros_undef.hpp>

#endif

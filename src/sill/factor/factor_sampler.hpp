#ifndef SILL_FACTOR_SAMPLER_HPP
#define SILL_FACTOR_SAMPLER_HPP

namespace sill {

  /**
   * A helper class that draws random samples from a marginal or
   * conditional distribution represented by a factor.
   * This is a dummy template that specifies the desired API only.
   * The factor classes need to specialize this template to perform
   * the actual computation.
   */
  template <typename F>
  class factor_sampler {
  public:
    //! The type that represents an ordered assignment in a compact format
    typedef typename F::index_type index_type;

    //! The vector of variables of F.
    //! This must be std::vector<typename F::variable_type*>
    typedef typename F::var_vector_type var_vector_type;

    //! Constructs the sampler for a marginal distribution
    factor_sampler(const F& factor);

    //! Constructs the sampler for a conditional distribution
    //! p(head | rest)
    factor_sampler(const F& factor, const var_vector_type& head);

    //! Draws a random sample from a marginal distribution
    template <typename RandomNumberGenerator>
    void operator()(index_type& sample, RandomNumberGenerator& rng) const;

    //! Draws a random sample from a conditional distribution
    template <typename RandomNumberGenerator>
    void operator()(index_type& head, const index_type& tail,
                    RandomNumberGenerator& rng) const;
    
  }; // class factor_evaluator

} // namespace sill

#endif

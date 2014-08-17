#ifndef SILL_FACTOR_RANDOM_CONCEPTS_HPP
#define SILL_FACTOR_RANDOM_CONCEPTS_HPP

namespace sill {

  /**
   * A concept for functor that generates marginal random factors.
   * \todo clarify how / when the parameter checking is performed.
   * \ingroup factor_random
   */
  template <typename RFG>
  struct RandomMarginalFactorGenerator {

    //! The domain type of the factors returned by this generator
    typedef typename RFG::domain_type domain_type;
    
    //! The type of factors returned by this generator
    typedef typename RFG::result_type result_type;

    //! The parameters that specify the distribution over the random factors
    typedef typename RFG::param_type param_type;

    //! Generates a marginal distribution over the specified variables
    template <typename RandomNumberGenerator>
    result_type operator()(const domain_type& args, RandomNumberGenerator& rng);

    //! Returns the parameter set associated with this generator
    param_type param() const;

    //! Sets the parameter set associated with this generator
    void param(const param_type& params);

  }; // concept RandomMarginalFactorGenerator

  /**
   * A concept for a functor that generates both marginal and conditional factors.
   * \ingroup factor_random
   */
  template <typename RFG>
  struct RandomFactorGenerator : public RandomMarginalFactorGenerator<RFG> {
    
    //! Generates a conditional distribution p(head | tail)
    template <typename RandomNumberGenerator>
    result_type operator()(const domain_type& head, const domain_type& tail,
                           RandomNumberGenerator& rng);

  }; // concept RandomFactorGenerator
  
  /**
   * A concept for a functor that generates random CRF factors.
   * Note: There are presently no classes in the library that implement this concept.
   * \ingroup factor_random
   */
  template <typename RFG>
  struct RandomCrfFactorGenerator {

    //! The output domain type of the factors returned by this generator
    typedef typename RFG::output_domain_type output_domain_type;
    
    //! The input domain type of the factors returned by this generator
    typedef typename RFG::input_domain_type input_domain_type;

    //! The type of factors returned by this generator
    typedef typename RFG::result_type result_type;

    //! The parameters that specify the distribution over the random factors
    typedef typename RFG::param_type param_type;

    //! Generates a conditional distribution p(head | tail)
    template <typename RandomNumberGenerator>
    result_type operator()(const output_domain_type& head,
                           const input_domain_type& tail,
                           RandomNumberGenerator& rng);

    //! Returns the parameter set associated with this generator
    param_type param() const;

    //! Sets the parameter set associated with this generator
    void param(const param_type& params);

  }; // concept RandomCrfFactorGenerator

}

#endif

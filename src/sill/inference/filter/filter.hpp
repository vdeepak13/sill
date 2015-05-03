#ifndef SILL_FILTER_HPP
#define SILL_FILTER_HPP

namespace sill {

  // Forward declarations
  template <typename F> class dynamic_bayesian_network;
  template <typename T> class domain;
  
  /**
   * An interface that represents a filter.
   * A filter can process observations over time, maintaining
   * a belief (posterior distribution) over the latests time steps
   * conditioned on all the observations up to now.
   *
   * \ingroup inference
   */
  template <typename F>
  class filter {
  public:
  public:
    // FactorizedInference types
    typedef typename F::real_type       real_type;
    typedef typename F::result_type     result_type;
    typedef typename F::variable_type   variable_type;
    typedef typename F::domain_type     domain_type;
    typedef typename F::assignment_type assignment_type;
    typedef F                           factor_type;
    typedef discrete_process<variable_type> process_type;

  public:
    //! Destructor
    virtual ~filter() {}

    //! Returns the dynamic Bayesian network associated with this filter
    virtual const dynamic_bayesian_network<F>& model() const = 0;

    //! Advances the state to the next time step
    virtual void advance() = 0;

    //! Multiplies in the likelihood to the belief state
    virtual void estimation(const F& likelihood) = 0;
    
    //! Extracts the beliefs over a subset of the processes
    virtual F belief(const domain<process_type*>& processes) const = 0;

    //! Extracts the beliefs over a subset of step-t variables
    virtual F belief(const domain_type& variables) const = 0;

    //! Extracts the belief over a single step-t variable
    virtual F belief(variable_type* v) const = 0;

  }; // class filter

} // namespace sill

#endif

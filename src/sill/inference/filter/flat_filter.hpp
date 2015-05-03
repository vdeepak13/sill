#ifndef SILL_FLAT_FILTER_HPP
#define SILL_FLAT_FILTER_HPP

#include <sill/inference/filter/filter.hpp>
#include <sill/model/dynamic_bayesian_network.hpp>

namespace sill {
  
  /**
   * A filter that represents the belief state as a factor.
   * \ingroup inference
   */
  template <typename F>
  class flat_filter : public filter<F> {
    
    // Public type declarations
    // =========================================================================
  public:
    typedef typename F::variable_type variable_type;
    typedef typename F::domain_type domain_type;
    typedef discrete_process<variable_type> process_type;

    // Private data members
    // =========================================================================
  private:
    //! The belief at time step t
    F belief_;

    //! The dynamic Bayesian network 
    dynamic_bayesian_network<F> dbn;

    //! A map that translates time-t+1 to time-t variables
    std::map<variable_type*, variable_type*> advance_var_map;

    // Public functions
    // =========================================================================
  public:
    //! Creates a filter for the given DBN.
    //! A copy of the DBN is stored inside this filter.
    flat_filter(const dynamic_bayesian_network<F>& dbn) 
      : dbn(dbn) {
      // the belief is simply the product of all factors at the first time step
      belief_ = prod_all(dbn.prior_model().factors());
      advance_var_map = 
        make_process_var_map(dbn.processes(), next_step, current_step);
    }

    //! Returns the current belief over all state variables
    const F& belief() const {
      return belief_;
    }

    // The documentation for the functions below is automatically copied
    // from the filter interface
    const dynamic_bayesian_network<F>& model() const {
      return dbn;
    }

    void advance() {
      // The simplistic approach: multiply in all of transition model
      // and eliminate the time-t variables
      // TODO: use variable elimination
      for (process_type* p : dbn.processes()) {
        belief_ *= dbn[p];
      }
      belief_ = belief_.marginal(variables(dbn.processes(), next_step));
      belief_.subst_args(advance_var_map);
    }
    
    void estimation(const F& likelihood) {
      assert(is_subset(likelihood.arguments(), belief_.arguments()));
      belief_ *= likelihood;
    }

    F belief(const std::set<process_type*>& processes) const {
      assert(is_subset(processes, dbn.processes()));
      return belief_.marginal(variables(processes, current_step));
    }

    F belief(const domain_type& variables) const {
      assert(is_subset(variables, belief_.arguments()));
      return belief_.marginal(variables);
    }

    F belief(variable_type* v) const {
      return belief(make_domain(v));
    }

  }; // class flat_filter

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

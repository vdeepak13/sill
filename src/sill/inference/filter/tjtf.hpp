#ifndef SILL_TJTF_HPP
#define SILL_TJTF_HPP

#include <sill/inference/interfaces.hpp>
#include <sill/model/dynamic_bayesian_network.hpp>

namespace sill {

  /**
   * A class that performs thin junction tree filtering. 
   * \todo citation
   */
  template <typename F>
  class tjtf : public filter<F> {
    concept_assert((DistributionFactor<F>));
    
    // Public type declarations
    // =========================================================================
  public:
    typedef typename F::variable_type variable_type;
    typedef typename F::domain_type domain_type;
    typedef discrete_process<variable_type> process_type;
    
    // Private data members
    // =========================================================================
  private:
    //! The belief at the current time step
    decomposable<F> belief_;

    //! The dynamic Bayesian network 
    dynamic_bayesian_network<F> dbn;

    //! The maximum permitted clique size
    size_t limit;

    //! A map that translates time-t+1 to time-t variables
    map<variable_type*, variable_type*> advance_var_map;
    
    // Public member functions
    // =========================================================================
  public:
    //! Creates a filter for the given DBN.
    //! A copy of the DBN is stored inside this filter.
    tjtf(const dynamic_bayesian_network<F>& dbn, size_t limit)
      : dbn(dbn), limit(limit) {
      // the belief is simply the product of all factors at the first time step
      belief_ *= dbn.prior().factors();
      advance_var_map = 
        make_process_var_map(dbn.processes(), next_step, current_step);
    }
    
    //! Returns the current belief over all state variables
    const decomposable<F>& belief() const {
      return belief_;
    }

    // The documentation for the functions below is automatically copied
    // from the filter interface
    const dynamic_bayesian_network<F>& model() const {
      return dbn;
    }

    void advance() {
      // Multiply in the transition models in topological order
      // and approximate after each one is incorporated
      std::vector<process_type*> procs = dbn.topological();
      for (process_type* p : procs) {
        belief_ *= dbn[p];
        belief_.thin(limit);
      }

      // Marginalize out the state variables from the previous time step
      for (process_type* p : procs) {
        belief_.marginalize_out(p->current());
        belief_.thin(limit);
      }
      
      // Rename step-t+1 variables to step-t
      belief_.subst_args(advance_var_map);
    }

    void estimation(const F& likelihood) {
      assert(likelihood.arguments().subset_of(belief_.arguments()));
      belief_ *= likelihood;
      belief_.thin(limit);
    }

    F belief(const set<process_type*>& processes) const {
      assert(processes.subset_of(dbn.processes()));
      return belief_.marginal(variables(processes, current_step));
    }

    F belief(const domain_type& variables) const {
      assert(variables.subset_of(belief_.arguments()));
      return belief_.marginal(variables);
    }
    
  }; // class tjtf

} // namespace sill

#endif


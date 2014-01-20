#ifndef SILL_BK98_FILTER_HPP
#define SILL_BK98_FILTER_HPP

#include <sill/model/junction_tree.hpp>
#include <sill/model/decomposable.hpp>
#include <sill/model/dynamic_bayesian_network.hpp>
#include <sill/inference/interfaces.hpp>
#include <sill/inference/variable_elimination.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Junction tree assumed density filtering. Implements the B&K 98
   * algorithm.  The filter takes as an input the DBN that specifies
   * the model and the assumed density, which is a junction tree with
   * cliques over the process variables at time step *t*, i.e.,
   * current_step.  If the parameter individual_marginals is set to
   * true, the filter performs the prediction/projection on each
   * clique separately; if individual_marginals is false, the filter
   * forms a joint distribution over t and t+1 and uses this distribution
   * to perform the prediction and projection.
   *
   * \todo citation
   * \ingroup inference
   */
  template <typename F>
  class bk98_filter : public filter<F> {

    // Public type declarations
    // =========================================================================
  public:
    typedef typename F::variable_type variable_type;
    typedef typename F::domain_type domain_type;
    typedef timed_process<variable_type> process_type;

    // Private data members
    // =========================================================================
  private:
    //! The dynamic Bayesian network 
    dynamic_bayesian_network<F> dbn;

    //! The approximation structure
    junction_tree<variable_type*> jt;

    //! If true, the clique marginals during advance() are computed individually
    bool individual_marginals;

    //! The belief at the current time step
    decomposable<F> belief_;

    //! A map that translates time-t+1 to time-t variables
    std::map<variable_type*, variable_type*> advance_var_map;
    
    // Public functions
    // =========================================================================
  public:
    //! Creates a filter for the given DBN.
    //! A copy of the DBN is stored inside this filter.
    bk98_filter(const dynamic_bayesian_network<F>& dbn,
                const junction_tree<variable_type*>& jt,
                bool individual_marginals)
      : dbn(dbn), jt(jt), individual_marginals(individual_marginals) {
      // the belief is simply the product of all factors at the first time step
      belief_ *= dbn.prior_model().factors();
      advance_var_map = 
        make_process_var_map(dbn.processes(), next_step, current_step);
      // we could also perform the projection at the first step
      // check the approximation structure
      foreach(const domain_type& clique, jt.cliques())
        check_index(clique, current_step);
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
      std::vector<F> marginals;
      
      if (individual_marginals) {
        // For each clique in the assumed density, collect all the CPDs
        // and the prior needed to compute the marginal over this clique
        foreach(size_t v, jt.vertices()) {
          domain_type clique_t1 = subst_index(jt.clique(v), next_step);
          domain_type ancestors = dbn.transition_model().ancestors(clique_t1);
          domain_type ancestors_t = intersect(ancestors, current_step);
          domain_type ancestors_t1 = intersect(ancestors, next_step);
          ancestors_t1.insert(clique_t1.begin(), clique_t1.end());
          decomposable<F> prior; belief_.marginal(ancestors_t, prior);
          std::vector<F> factors(prior.factors().begin(),prior.factors().end());
          foreach(variable_type* v, ancestors_t1)
            factors.push_back(dbn[v]);
          
//           using namespace std;
//           size_t maxsize = 0;
//           for(size_t i = 0; i < factors.size(); i++) 
//             maxsize = std::max(maxsize, factors[i].arguments().size());
//           decomposable<F> joint; joint *= factors;
//           cout << "clique " << jt.clique(v) << ": "
//                << "factor size " << maxsize << ", "
//                << "tree width " << joint.tree_width() << endl;
          
          F marginal = variable_elimination(factors, clique_t1, sum_product<F>(),
                                            min_degree_strategy());
          marginal.subst_args(advance_var_map);
          marginals.push_back(marginal);
        }
      } else {
        // Form the joint over the two time steps and extract the marginals
        std::list<F> factors;
        foreach(process_type* p, dbn.processes())
          factors.push_back(dbn[p]); // should be called dbn.transition(p);
        belief_ *= factors;
        foreach(size_t v, jt.vertices()) {
          domain_type clique_t1 = subst_index(jt.clique(v), next_step);
          F marginal = belief_.marginal(clique_t1);
          marginal.subst_args(advance_var_map);
          marginals.push_back(marginal);
        }
      }
//       using namespace std;
//       cerr << "Initializing with jt " << jt << ", marginals " << marginals 
//            << endl;

      // Construct a decomposable model with the given structure and marginals
      belief_.initialize(jt, marginals);
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

  }; // class bk98_filter

}; // namespace sill

#include <sill/macros_undef.hpp>

#endif

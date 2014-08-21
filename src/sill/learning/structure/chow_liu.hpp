#ifndef SILL_CHOW_LIU_HPP
#define SILL_CHOW_LIU_HPP

#include <set>

#include <sill/iterator/transform_output_iterator.hpp>
#include <sill/learning/factor_mle/factor_mle.hpp>
#include <sill/model/decomposable.hpp>
#include <sill/model/projections.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Class for learning the Chow-Liu tree over variables X in the given dataset.
   * Models the Learner concept.
   *
   * @tparam F  type of factor for the model
   * \ingroup learning_structure
   * \see Learner
   */
  template <typename F>
  class chow_liu {
  public:
    // Learner concept types
    typedef typename F::real_type                real_type;
    typedef decomposable<F>                      model_type;
    typedef typename factor_mle<F>::dataset_type dataset_type;
    typedef typename factor_mle<F>::param_type   param_type;

    // Other public types
    typedef typename F::variable_type    variable_type;
    typedef typename F::domain_type      domain_type;
    typedef typename F::var_vector_type  var_vector_type;
    typedef typename F::marginal_fn_type marginal_fn_type;

    // Public methods
    // =========================================================================
  public:
    /**
     * Constructs the Chow-Liu learner over the given argument set.
     */
    chow_liu(const domain_type& dom)
      : vars(dom.begin(), dom.end()) { }

    /**
     * Learns a decomposable model using the default parameters.
     */
    real_type learn(const dataset_type& ds, model_type& model) const {
      return learn(factor_mle<F>(&ds), model);
    }

    /**
     * Learns a decomposable model for the given dataset and parameters.
     */
    real_type learn(const dataset_type& ds,
                    const param_type& params,
                    model_type& model) const {
      return learn(factor_mle<F>(&ds, params), model);
    }

    /**
     * Learns a decomposable model from the marginals provided by the given
     * functor.
     */
    real_type learn(marginal_fn_type estim,
                    model_type& model,
                    std::map<domain_type, real_type>* edge_score_map = NULL) const {
      if (vars.empty()) {
        return 0.0;
      }
    
      // g will hold weights (mutual information) and factors F for each edge.
      typedef std::pair<double, F> edge_mi_pot;
      typedef undirected_graph<variable_type*, void_, edge_mi_pot> ig_type;
      ig_type g;
      foreach(variable_type* v, vars) {
        g.add_vertex(v);
      }
      for (size_t i = 0; i < vars.size() - 1; ++i) {
        for (size_t j = i+1; j < vars.size(); ++j) {
          domain_type edge_dom = make_domain(vars[i], vars[j]);
          F f = estim(edge_dom);
          double mi = f.mutual_information(make_domain(vars[i]),
                                           make_domain(vars[j]));
          g.add_edge(vars[i], vars[j], std::make_pair(mi, f));
          if (edge_score_map) {
            edge_score_map->insert(std::make_pair(edge_dom, mi));
          }
        }
      }

      // Create a MST over the graph g.
      std::vector<typename ig_type::edge> edges;
      kruskal_minimum_spanning_tree
        (g, std::back_inserter(edges), impl::mst_weight_functor<F>(g));

      // Extract the objective value and factors
      real_type sum_mi = 0.0;
      std::vector<F> mst_factors;
      foreach(typename ig_type::edge e, edges) {
        sum_mi += g[e].first;
        mst_factors.push_back(g[e].second);
      }

      // Create a decomposable model consisting of the cliques in edges
      model.initialize(mst_factors);
      return sum_mi;
    }

    // Private data
    // =========================================================================
  private:
    //! The vector variables in the learned model
    var_vector_type vars;

  }; // class chow_liu

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

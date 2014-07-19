#ifndef SILL_CHOW_LIU_HPP
#define SILL_CHOW_LIU_HPP

#include <set>

#include <sill/iterator/transform_output_iterator.hpp>
#include <sill/learning/parameter/factor_learner.hpp>
#include <sill/model/decomposable.hpp>
#include <sill/model/projections.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Class for learning the Chow-Liu tree over variables X in the given dataset.
   *
   * @tparam F  type of factor for model
   * \ingroup learning_structure
   */
  template <typename F>
  class chow_liu {

    // Public types
    // =========================================================================
    typedef typename F::variable_type   variable_type;
    typedef typename F::domain_type     domain_type;
    typedef typename F::var_vector_type var_vector_type;

    // Private data
    // =========================================================================
  private:
    //! Learned tree
    decomposable<F> model_;

    //! Map: (pair of variables)-->(mutual information score from Chow-Liu)
    std::map<domain_type, double> edge_score_mapping_;

    // Public methods
    // =========================================================================
  public:

    /**
     * Constructor which learns a Chow-Liu tree using the given base learner.
     * @param args variables over which to learn a tree.
     * @param flearn returns the learned marginals for a subset of variables
     */
    chow_liu(const domain_type& dom, const factor_learner<F>& flearn) {
      if (dom.empty()) {
        return;
      }

      var_vector_type vars(dom.begin(), dom.end());

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
          F f = flearn(edge_dom);
          double mi = f.mutual_information(make_domain(vars[i]),
                                           make_domain(vars[j]));
          g.add_edge(vars[i], vars[j], std::make_pair(mi, f));
          edge_score_mapping_[edge_dom] = mi;
        }
      }

      // Create a MST over the graph g.
      std::vector<F> mst_factors;
      kruskal_minimum_spanning_tree
        (g, transformed_output(back_inserter(mst_factors),
                               impl::mst_edge2f_functor<F>(g)),
         impl::mst_weight_functor<F>(g));

      // Create a decomposable model consisting of the cliques in mst_edges
      model_ *= mst_factors;
    }

    //! Return the learned tree.
    const decomposable<F>& model() const {
      return model_;
    }

    //! Map: (pair of variables)-->(mutual information score from Chow-Liu).
    //! (This info is retained if set to do so in the parameters.)
    const std::map<typename F::domain_type, double>& edge_score_mapping() const{
      return edge_score_mapping_;
    }

  }; // class chow_liu

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

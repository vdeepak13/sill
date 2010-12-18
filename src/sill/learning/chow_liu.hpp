
#ifndef SILL_CHOW_LIU_HPP
#define SILL_CHOW_LIU_HPP
#include <set>

#include <sill/learning/crossval_parameters.hpp>
#include <sill/learning/dataset/dataset.hpp>
#include <sill/learning/learn_factor.hpp>
#include <sill/model/decomposable.hpp>
#include <sill/model/free_functions.hpp>

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

    //    concept_assert((LearnableDistributionFactor<F>));

    // Public types
    // =========================================================================
  public:

    struct parameters {

      /**
       * Regularization.  See the factor type's marginal() method for details.
       * (If < 0, resort to factor defaults.)
       *  (default = -1)
       */
      double lambda;

      //! If true, retain the edge_score_mapping: (pair of variables)-->(score).
      //!  (default = false)
      bool retain_edge_score_mapping;

      //! If true, do cross-validation to choose the regularization lambda
      //! used to estimate the marginal for each edge, using the below
      //! parameters n_folds, minvals, maxvals, nlambdas, and zoom.
      //!  (default = false)
      bool do_cv;

      //! Parameters specifying how to do cross-validation (if do_cv == true).
      // TO DO: THESE NEED TO BE OF THE DIMENSION REQUIRED BY THE FACTOR TYPE, BUT FACTORS DO NOT IMPLEMENT CV YET.
      //crossval_params<

      parameters()
        : lambda(-1), retain_edge_score_mapping(false), do_cv(false) { }

      bool valid() const {
        if (do_cv)
          return false; // NOT YET IMPLEMENTED!
        return true;
      }

    }; // struct parameters

    // Private data
    // =========================================================================
  private:

    parameters params;

    //! Learned tree
    decomposable<F> model_;

    //! Map: (pair of variables)-->(mutual information score from Chow-Liu)
    std::map<typename F::domain_type, double> edge_score_mapping_;

    // Public methods
    // =========================================================================
  public:

    /**
     * Constructor which learns a Chow-Liu tree from the given dataset.
     * @param X             Variables over which to learn a tree.
     * @param ds            Dataset to use for computing marginals.
     */
    chow_liu(const forward_range<typename F::variable_type*>& X_,
             const dataset& ds, const parameters& params = parameters())
      : params(params) {

      typedef typename F::variable_type variable_type;
      assert(ds.size() > 0);
      std::vector<variable_type*> X(X_.begin(), X_.end());
      if (X.size() == 0)
        return;

      // g will hold weights (mutual information) and factors F for each edge.
      typedef std::pair<double, F> edge_mi_pot;
      typedef undirected_graph<variable_type*, void_, edge_mi_pot> ig_type;
      ig_type g;
      foreach(variable_type* v, X)
        g.add_vertex(v);
      for (size_t i(0); i < X.size() - 1; ++i) {
        for (size_t j(i+1); j < X.size(); ++j) {
          typename F::domain_type
            edge_dom(make_domain<variable_type>(X[i],X[j]));
          F f((params.lambda < 0 ?
               learn_marginal<F>(edge_dom, ds) :
               learn_marginal<F>(edge_dom, ds, params.lambda)));
          double mi(f.mutual_information(make_domain(X[i]), make_domain(X[j])));
          g.add_edge(X[i], X[j], std::make_pair(mi, f));
          if (params.retain_edge_score_mapping) {
            edge_score_mapping_[edge_dom] = mi;
          }
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

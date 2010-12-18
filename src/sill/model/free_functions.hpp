
#ifndef SILL_MODEL_FREE_FUNCTIONS_HPP
#define SILL_MODEL_FREE_FUNCTIONS_HPP

#include <cmath>

#include <boost/range/iterator_range.hpp>

#include <sill/factor/table_factor.hpp>
#include <sill/graph/undirected_graph.hpp>
#include <sill/model/bayesian_network.hpp>
#include <sill/model/decomposable.hpp>
#include <sill/model/markov_network.hpp>
#include <sill/iterator/transform_output_iterator.hpp>

#include <sill/range/concepts.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Create a pairwise Markov network from a factorized model.
   * If the factorized model contains factors which are over more than 2
   * variables, then this unrolls the factors to create an equivalent pairwise
   * Markov network as follows:
   *  - Given a factor over ABC (say each are binary),
   *  - Create a factor over a new variable D with 8 values corresponding to
   *    those in the original factor.
   *  - Add an edge potential between A and D which enforces the constraint that
   *    A = 0 iff D has a value which corresponds to A having value 0.  Do the
   *    same for B and C.
   *  - This does not unroll the potential if it is only over one variable.
   *
   * @param fm factorized model to be transformed into a pairwise Markov net
   * @param u  universe used to create new variables for the Markov net
   * @return pair:
   *          - pairwise Markov network
   *          - mapping from the new variables in the Markov network to
   *            the (ordered) sets of old variables the new variables represent
   * @see restore_bp_pairwise_markov_network
   *
   * \todo Extend this so that it works with any types of factor.  This could be
   *       done by creating functions for each type of factor for creating
   *       indicator potentials of the type needed here.
   */
  std::pair
  <pairwise_markov_network<table_factor >,
   std::map<finite_variable*, std::vector<finite_variable*> > >
  fm2pairwise_markov_network
  (const factorized_model<table_factor >& fm, universe& u);

  /**
   * Convert the node beliefs from the given belief propagation engine (over
   * a pairwise Markov net which represents a general Markov net
   * (see fm2pairwise_markov_network())) into a set of factors which are only
   * over the original variables (in the general Markov net).
   * @param engine      BP engine over pairwise Markov net
   * @param orig_vars   variables in original general Markov net
   * @param var_mapping mapping from temp variables in pairwise net to (ordered)
   *                    sets of variables in orig_vars
   * @return set of factors (beliefs) over orig_vars
   * @see fm2pairwise_markov_network
   */
  template <typename F, typename FactorRange>
  std::vector<F> restore_unrolled_markov_network
  (const FactorRange& node_beliefs, const domain& orig_vars,
   std::map<finite_variable*, std::vector<finite_variable*> >& var_mapping) {
    concept_assert((InputRangeConvertible<FactorRange, F>));
    using namespace sill;
    std::vector<F> fvec;
    foreach(const F& f, node_beliefs) {
      finite_variable* v = *(f.arguments().begin());
      if (orig_vars.count(v))
        fvec.push_back(f);
      else
        fvec.push_back(f.roll_up(var_mapping[v]));
    }
    return fvec;
  }
  
  namespace impl {

    //! For Chow-Liu.
    template <typename F>
    class mst_weight_functor {
    public:
      typedef std::pair<double, F> edge_mi_pot;
      typedef undirected_graph<typename F::variable_type*, void_, edge_mi_pot>
        ig_type;
      typedef typename ig_type::edge argument_type;
      typedef double result_type;
    private:
      const ig_type* g;
    public:
      mst_weight_functor(const ig_type& g) : g(&g) { }
      double operator()(typename ig_type::edge e) const {
        return -(g->operator[](e).first);
      }
    };

    //! For Chow-Liu.
    template <typename F>
    class mst_edge2f_functor {
    public:
      typedef std::pair<double, F> edge_mi_pot;
      typedef undirected_graph<typename F::variable_type*, void_, edge_mi_pot>
        ig_type;
      typedef F result_type;
    private:
      const ig_type* g;
    public:
      mst_edge2f_functor(const ig_type& g) : g(&g) { }
      F operator()(typename ig_type::edge e) const {
        return g->operator[](e).second;
      }
    };

  } // namespace impl

  /**
   * Convert a range of beliefs (normalized factors from belief propagation)
   * into a decomposable model (of treewidth 1) by
   * choosing a subtree of the implicit Markov net.
   * Note: This assumes that BP has converged so that the beliefs are consistent
   * in all subtrees which satisfy the running intersection property.
   *
   * To choose the subtree, this function computes the mutual information
   * between all pairs of variables appearing in factors together and
   * then build a maximum weight spanning tree over those variables.
   */
  template <typename F, typename FactorRange>
  decomposable<F> factors2thin_decomposable(const FactorRange& factors) {
    concept_assert((InputRangeConvertible<FactorRange, F>));
    using namespace sill;
    /*
      Create a graph and a mapping of edges to weights.
      - for each belief
        - if the belief is only over 1 variable, store it in a separate list;
          go to the next belief
        - otherwise, add variable nodes to graph as necessary
        - for each pair of variables in the belief
          - create an edge between the nodes if necessary
          - marginalize the belief to make it only over that pair,
            and compute the mutual information between the variables
          - if the edge has no potential or the current one has lower mutual
            information between the variables, make the marginalized belief
            the new edge potential
    */
    typedef std::pair<double, F> edge_mi_pot;
    typedef undirected_graph<finite_variable*, void_, edge_mi_pot> ig_type;
    ig_type g;

    // List of single-variable beliefs
    std::vector<F> single_var_beliefs;
    foreach(const F& f, factors) {
      if (f.arguments().size() == 1)
        single_var_beliefs.push_back(f);
      else if (f.arguments().size() > 1) {
        std::vector<finite_variable*> args(f.arguments().begin(),
                                           f.arguments().end());
        foreach(finite_variable* v, args)
          if (!(g.contains(v)))
            g.add_vertex(v);
        for (size_t i = 0; i < args.size(); i++) {
          for (size_t j = i + 1; j < args.size(); j++) {
            F edge_f(f.marginal(make_domain(args[i], args[j])));
            // TODO: Once factors implement mutual info, replace this code here.
            double mi(0);
            F edge_f_i(edge_f.marginal(make_domain(args[i])));
            F edge_f_j(edge_f.marginal(make_domain(args[j])));
            foreach(finite_assignment a, edge_f.assignments())
              mi += edge_f.v(a) * (edge_f.logv(a) -
                                 edge_f_i.logv(a) - edge_f_j.logv(a));
            // Get or make edge
            if (g.contains(args[i], args[j]) ) {
              typename ig_type::edge e(g.get_edge(args[i], args[j]));
              // compare MI to see which potential we should use
              if (g[e].first < mi)
                g[e] = std::make_pair(mi, edge_f);
            } else {
              g.add_edge(args[i], args[j], std::make_pair(mi, edge_f));
              /*
              typename ig_type::edge e(g.get_edge(args[i], args[j]));
              bool edge_found;
              boost::tie(e, edge_found) = g.add_edge(args[i], args[j]);
              assert(edge_found);
              g[e] = std::make_pair(mi, edge_f);
              */
            }
          }
        }
      }
    }
    // Create a MST over the graph g.
    std::vector<F> mst_factors;
    kruskal_minimum_spanning_tree
      (g, transformed_output(back_inserter(mst_factors),
                             impl::mst_edge2f_functor<F>(g)),
       impl::mst_weight_functor<F>(g));
    // Insert the single-variable beliefs which are not already represented by
    // 2-variable beliefs.
    domain decomposable_args(boost::begin(g.vertices()), boost::end(g.vertices()));
    foreach(F& f, single_var_beliefs) {
      finite_variable* f_arg = *(f.arguments().begin());
      if (!(decomposable_args.count(f_arg))) {
        decomposable_args.insert(f_arg);
        mst_factors.push_back(f);
      }
    }
    // Create a decomposable model consisting of the cliques in mst_edges
    decomposable<F> model(mst_factors);
    return model;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_MODEL_FREE_FUNCTIONS_HPP

#ifndef SILL_MODEL_PROJECTIONS_HPP
#define SILL_MODEL_PROJECTIONS_HPP

#include <sill/graph/undirected_graph.hpp>
#include <sill/iterator/transform_output_iterator.hpp>
#include <sill/model/decomposable.hpp>
#include <sill/range/concepts.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  namespace impl {

    //! For Chow-Liu: functor for getting the negated weight of an edge.
    template <typename F>
    class mst_weight_functor {
    public:
      typedef std::pair<double, F> edge_mi_pot;
      typedef undirected_graph<typename F::variable_type, void_, edge_mi_pot>
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

    //! For Chow-Liu: functor for getting the factor associated with an edge.
    template <typename F>
    class mst_edge2f_functor {
    public:
      typedef std::pair<double, F> edge_mi_pot;
      typedef undirected_graph<typename F::variable_type, void_, edge_mi_pot>
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
   * then builds a maximum weight spanning tree over those variables.
   */
  template <typename F, typename FactorRange>
  decomposable<F> factors2thin_decomposable(const FactorRange& factors) {
    concept_assert((InputRangeConvertible<FactorRange, F>));
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
    typedef typename F::variable_type   variable_type;
    typedef std::pair<double, F> edge_mi_pot;
    typedef undirected_graph<variable_type, void_, edge_mi_pot> ig_type;
    ig_type g;

    // List of single-variable beliefs
    std::vector<F> single_var_beliefs;
    foreach(const F& f, factors) {
      if (f.arguments().size() == 1)
        single_var_beliefs.push_back(f);
      else if (f.arguments().size() > 1) {
        std::vector<variable_type> args(f.arguments().begin(),
                                           f.arguments().end());
        foreach(variable_type v, args)
          if (!(g.contains(v)))
            g.add_vertex(v);
        for (size_t i = 0; i < args.size(); i++) {
          for (size_t j = i + 1; j < args.size(); j++) {
            F edge_f(f.marginal(make_domain(args[i], args[j])));
            double mi = edge_f.mutual_information(make_domain(args[i]),
                                                  make_domain(args[j]));
            // Get or make edge
            if (g.contains(args[i], args[j]) ) {
              typename ig_type::edge e(g.get_edge(args[i], args[j]));
              // compare MI to see which potential we should use
              if (g[e].first < mi)
                g[e] = std::make_pair(mi, edge_f);
            } else {
              g.add_edge(args[i], args[j], std::make_pair(mi, edge_f));
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
    domain decomposable_args(boost::begin(g.vertices()),
                             boost::end(g.vertices()));
    foreach(F& f, single_var_beliefs) {
      variable_type f_arg = *(f.arguments().begin());
      if (!(decomposable_args.count(f_arg))) {
        decomposable_args.insert(f_arg);
        mst_factors.push_back(f);
      }
    }
    // Create a decomposable model consisting of the cliques in mst_edges
    decomposable<F> model(mst_factors);
    return model;
  } // factors2thin_decomposable

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_MODEL_PROJECTIONS_HPP

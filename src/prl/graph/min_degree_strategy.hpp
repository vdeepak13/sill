#ifndef PRL_MIN_DEGREE_STRATEGY_HPP
#define PRL_MIN_DEGREE_STRATEGY_HPP

#include <prl/graph/concepts.hpp>
#include <prl/range/algorithm.hpp>

namespace prl {

  /**
   * Min-degree elimination strategy.
   * This type models the EliminationStrategy concept.
   * \ingroup graph_types
   */
  struct min_degree_strategy {

    /**
     * The priority type associated with each vertex.
     */
    typedef int priority_type;

    /**
     * Computes the priority of a vertex, which is its negative degree.
     * This makes nodes with smaller degrees have higher priority.
     */
    template <typename Graph>
    int priority(typename Graph::vertex v, const Graph& g) {
      //concept_assert((IncidenceGraph<Graph>));
      return -static_cast<int>(g.out_degree(v));
    }

    /**
     * Computes the set of vertices whose priority may change if a
     * designated vertex is eliminated.  For the min-degree strategy,
     * this is the neighbors of the eliminated vertex.
     *
     * Here, we use OutIt (OutputIterator), rather than returning a transformed
     * range, since the list of updated nodes will be used after making changes
     * to the graph (which would invalidate the transformed range).
     */
    template <typename Graph, typename OutIt>
    void updated(typename Graph::vertex v, const Graph& g, OutIt updated) {
      //concept_assert((IncidenceGraph<Graph>));
      //concept_assert((OutputIterator<OutIt, typename Graph::vertex>));
      prl::copy(g.neighbors(v), updated);
    }

  }; // struct min_degree_strategy

} // namespace prl

#endif

#ifndef PRL_MIN_FILL_STRATEGY_HPP
#define PRL_MIN_FILL_STRATEGY_HPP

#include <set>

#include <boost/range/metafunctions.hpp>

#include <prl/global.hpp>
#include <prl/range/algorithm.hpp>

namespace prl {

  /**
   * Represents a min-fill elimination strategy.
   * This type models the EliminationStrategy concept.
   * \ingroup graph_types
   */
  struct min_fill_strategy {

    //! The priority type associated with each vertex.
    typedef int priority_type;

    /**
     * Computes the priority of a vertex, which is the negative of the
     * number of fill edges caused by its elimination.
     * This makes nodes with smaller fill-in have higher priority.
     */
    template <typename Graph>
    int priority(typename Graph::vertex v, const Graph& g) {
      //concept_assert((IncidenceGraph<G>));

      int n = 0;
      typename Graph::neighbor_iterator begin, end;
      for (boost::tie(begin, end) = g.neighbors(v); begin != end; ++begin) {
        typename Graph::neighbor_iterator cur = begin;
        while (++cur != end) {
          //if (!g.edge(*cur, *begin).second) ++n;
          if (!g.contains(*begin, *cur)) ++n;
        }
      }
      return -n;
    }

    /**
     * Computes the set of vertices whose priority may change if a
     * designated vertex is eliminated.
     * For the min-fill strategy, this is the neighbors of the eliminated
     * vertex, as well as the neighbors' neighbors.
     */
    template <typename Graph, typename OutIt>
    void updated(typename Graph::vertex v, const Graph& g, OutIt updated) {
      //concept_assert((IncidenceGraph<Graph>));
      //concept_assert((OutputIterator<OutIt, typename Graph::vertex>));
      std::set<typename Graph::vertex> update_set;
      // It is faster to store the values in a set than to output them 
      // multiple times (which causes further priority updates)

      typename Graph::neighbor_iterator begin, end;
      typename Graph::neighbor_iterator n_begin, n_end;
      for (boost::tie(begin, end) = g.neighbors(v); begin != end; ++begin) {
        typename Graph::vertex nbr = *begin;
        update_set.insert(nbr);
        for (boost::tie(n_begin, n_end) = g.neighbors(nbr);
             n_begin != n_end; ++n_begin) {
          if (*n_begin == v) continue;
          update_set.insert(*n_begin);
        }
      }
      prl::copy(update_set, updated);
    }

  }; // struct min_fill_strategy

} // namespace prl

#endif

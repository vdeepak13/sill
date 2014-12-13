#ifndef SILL_GRAPH_MST_HPP
#define SILL_GRAPH_MST_HPP

#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <boost/property_map/property_map.hpp>

#include <sill/graph/algorithm/index_map.hpp>
#include <sill/graph/algorithm/functor_property_map.hpp>
#include <sill/graph/algorithm/vertex_index.hpp>
#include <sill/stl_concepts.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Kruskal Minimum Spanning Tree (MST) algorithm.
   * 
   * @param g  graph
   * @param spanning_tree_edges  iterator into which the edges of the MST
   *                             are inserted
   * \ingroup graph_algorithms
   */
  template <typename Graph, typename OutIt>
  void kruskal_minimum_spanning_tree(const Graph& g, OutIt spanning_tree_edges){
    concept_assert((OutputIterator<OutIt, typename Graph::edge>));
    boost::unordered_map<typename Graph::vertex, size_t> map;
    sill::vertex_index(g, map);
    boost::kruskal_minimum_spanning_tree
      (g,
       spanning_tree_edges,
       boost::vertex_index_map(boost::make_assoc_property_map(map)));
  }

  /**
   * Kruskal Minimum Spanning Tree (MST) algorithm.
   * 
   * @param g  graph
   * @param spanning_tree_edges  iterator into which the edges of the MST
   *                             are inserted
   * @param f  functor which returns the weight of each edge
   * \ingroup graph_algorithms
   */
  template <typename Graph, typename OutIt, typename F>
  void kruskal_minimum_spanning_tree(const Graph& g, 
                                     OutIt spanning_tree_edges, F f) {
    concept_assert((OutputIterator<OutIt, typename Graph::edge>));
    boost::unordered_map<typename Graph::vertex, size_t> map;
    sill::vertex_index(g, map);
    boost::kruskal_minimum_spanning_tree
      (g,
       spanning_tree_edges,
       boost::vertex_index_map(boost::make_assoc_property_map(map)).
       weight_map(make_functor_property_map(f)));
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

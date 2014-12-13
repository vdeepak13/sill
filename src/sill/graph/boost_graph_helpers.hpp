#ifndef BOOST_GRAPH_HELPERS_HPP
#define BOOST_GRAPH_HELPERS_HPP

/**
 * \file boost_graph_helper.hpp
 * This file contains a number of generic functions that convert BGL's 
 * free function calls to PRL graph member function calls.
 */

#include <utility>

#include <sill/macros_def.hpp>

namespace sill {
  
  // Not much magic here.. it's all pretty repetitive

  // Access
  /////////
  template <typename G>
  std::pair<typename G::vertex_iterator,
	    typename G::vertex_iterator> 
  vertices(const G& g, typename G::vertex* = 0) {
    return g.vertices();
  }    

  template <typename G>
  std::pair<typename G::edge_iterator,
	    typename G::edge_iterator>
  edges(const G& g, typename G::vertex* = 0) {
    return g.edges();
  }

  template <typename G>
  std::pair<typename G::adjacency_iterator,
	    typename G::adjacency_iterator>
  adjacent_vertices(typename G::vertex v, const G& g) {
    return g.adjacent_vertices(v);
  }

  template <typename G>
  std::pair<typename G::inv_adjacency_iterator, 
	    typename G::inv_adjacency_iterator>
  inv_adjacent_vertices(typename G::vertex v, const G& g) {
    return g.inv_adjacent_vertices(v);
  }

  template <typename G>
  std::pair<typename G::out_edge_iterator, 
	    typename G::out_edge_iterator>
  out_edges(typename G::vertex v, const G& g) {
    return g.out_edges(v);
  }

  template <typename G>
  std::pair<typename G::in_edge_iterator, 
	    typename G::in_edge_iterator>
  in_edges(typename G::vertex v, const G& g) {
    return g.in_edges(v);
  }

  template <typename G>
  typename G::vertex source(typename G::edge e, const G& g) {
    return e.source();
  }

  template <typename G>
  typename G::vertex target(typename G::edge e, const G& g) {
    return e.target();
  }

  template <typename G>
  size_t out_degree(typename G::vertex v, const G& g){
    return g.out_degree(v);
  }

  template <typename G>
  size_t in_degree(typename G::vertex v, const G& g) {
    return g.in_degree(v);
  }

  template <typename G>
  size_t num_vertices(const G& g, typename G::vertex* = 0) {
    return g.num_vertices();
  }

  template <typename G>
  size_t num_edges(const G& g, typename G::vertex* = 0) {
    return g.num_edges();
  }

  template <typename G>
  std::pair<typename G::edge, bool>
  edge(typename G::vertex u, typename G::vertex v, const G& g) {
    return g.get_edge(u, v);
  }

  // Modification
  ///////////////
  template <typename G>
  typename G::vertex add_vertex(G& g, typename G::vertex* = 0) {
    BOOST_STATIC_ASSERT(sizeof(G)==0); // unsupported function
  }

  template <typename G, typename VertexProperty>
  typename G::vertex 
  add_vertex(const VertexProperty& p, G& g, typename G::vertex* = 0) {
    BOOST_STATIC_ASSERT(sizeof(G)==0); // unsupported function
  }

  template <typename G>
  void clear_vertex(typename G::vertex v, G& g) {
    g.clear_vertex(v);
  }

  template <typename G>
  void clear_out_edges(typename G::vertex u, G& g) {
    g.clear_edges(u);
  }

  template <typename G>
  void clear_in_edges(typename G::vertex v, G& g) {
    g.clear_in_edges(v);
  }

  template <typename G>
  void remove_vertex(typename G::vertex v, G& g) {
    g.remove_vertex(v);
  }

  template <typename G>
  std::pair<typename G::edge, bool>
  add_edge(typename G::vertex u, typename G::vertex v, G& g) {
    return g.add_edge(u, v); // FIXME
  }

  template <typename G, typename EdgeProperty>
  std::pair<typename G::edge, bool>
  add_edge(typename G::vertex u, typename G::vertex v,
           const EdgeProperty& p, G& g) {
    return g.add_edge(u, v, p);
  }

  template <typename G>
  void remove_edge(typename G::vertex u, typename G::vertex v, G& g) {
    g.remove_edge(u, v);
  }


  template <typename G>
  void remove_edge(typename G::edge e, G& g, typename G::bgl* = 0) {
    g.remove_edge(e);
  }

  template <typename G>
  void remove_edge(typename G::out_edge_iterator it, G& g, 
                   typename G::vertex* =0){
    g.remove_edge(it);
  }

  template <typename G, typename Predicate>
  void remove_edge_if(typename G::vertex u, Predicate p, G& g) {
    g.remove_edge_if(u, p);
  }

  template <typename G, typename Predicate>
  void remove_out_edge_if(typename G::vertex u, Predicate p, G& g) {
    g.remove_out_edge_if(u, p);
  }

  template <typename G, typename Predicate>
  void remove_in_edge_if(typename G::vertex u, Predicate p, G& g) {
    g.remove_in_edge_if(u, p);
  }

//   // property maps
//   template <typename G>
//   map<typename G::vertex, size_t>
//   get(boost::vertex_index_t&, const G& g, typename G::vertex* = 0) {
//     return make_index_map(g.vertices());
//     // BOOST_STATIC_ASSERT(sizeof(G)==0); // not implemented yet
//   }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

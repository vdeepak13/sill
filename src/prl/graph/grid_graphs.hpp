#ifndef PRL_GRID_GRAPHS_HPP
#define PRL_GRID_GRAPHS_HPP

//! \todo this file should be renamed to grid_graph

#include <boost/multi_array.hpp>
#include <prl/graph/undirected_graph.hpp>

#include <prl/macros_def.hpp>

// Constructs some common graphical models

namespace prl {
  
  //! Creates a grid graph and a map of the corresponding vertices
  //! \ingroup graph_special
  template <typename VP, typename EP>
  boost::multi_array<size_t,2>
  make_grid_graph(size_t m, size_t n, undirected_graph<size_t, VP, EP>& g) {
    typedef undirected_graph<size_t, VP, EP> graph_type;
    size_t ind = 1;
    
    // create the vertices
    boost::multi_array<size_t, 2> 
    vertex(boost::extents[m][n]);
    for(size_t i = 0; i < m; i++)
      for(size_t j = 0; j < n; j++) {
        vertex[i][j] = ind; g.add_vertex(ind++);
      }
    
    // create the edges
    for(size_t i = 0; i < m; i++) {
      for(size_t j = 0; j < n; j++) {
        if (j < n-1) g.add_edge(vertex[i][j], vertex[i][j+1]); 
        if (i < m-1) g.add_edge(vertex[i][j], vertex[i+1][j]);
      }
    }
    
    return vertex;
  }

  //! Creates a grid graph and a map of the corresponding vertices
  //! \ingroup graph_special
  template <typename Graph>
  boost::multi_array<typename Graph::vertex,2>
  make_grid_graph(size_t m, size_t n, Graph& g,
                  const std::vector<typename Graph::vertex>& vertices) {
    assert(vertices.size() == m*n);
    size_t k = 0;
    
    // create the vertices
    boost::multi_array<typename Graph::vertex, 2> vertex(boost::extents[m][n]);
    for(size_t i = 0; i < m; i++)
      for(size_t j = 0; j < n; j++) {
        g.add_vertex(vertices[k]);
        vertex[i][j] = vertices[k];
        k++;
      }
    
    // create the edges
    for(size_t i = 0; i < m; i++) {
      for(size_t j = 0; j < n; j++) {
        if (j < n-1) g.add_edge(vertex[i][j], vertex[i][j+1]); 
        if (i < m-1) g.add_edge(vertex[i][j], vertex[i+1][j]);
      }
    }
    
    return vertex;
  }
  
}

#include <prl/macros_undef.hpp>

#endif

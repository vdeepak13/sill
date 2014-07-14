#define BOOST_TEST_MODULE triangulation
#include <boost/test/unit_test.hpp>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#include <sill/graph/min_fill_strategy.hpp>
//#include <sill/graph/min_degree_strategy.hpp>
#include <sill/graph/triangulation.hpp>
#include <sill/graph/undirected_graph.hpp>

#include <iostream>

#include <sill/macros_def.hpp>

BOOST_AUTO_TEST_CASE(test_coverage) {
  using namespace sill;
  size_t nvertices = 20;
  size_t nedges = 80;
    
  // create a random graph with given number of vertices and edges
  undirected_graph<int> g;
  boost::mt19937 rng;
  boost::random::uniform_int_distribution<> dis(1, nvertices);
  for (size_t i = 0; i < nvertices; ++i) {
    g.add_vertex(i+1);
  }
  for (size_t i = 0; i < nedges; ++i) {
    g.add_edge(dis(rng), dis(rng));
  }
  
  // triangulate
  typedef std::set<int> node_set;
  std::vector<node_set> cliques;
  triangulate(g, std::back_inserter(cliques), min_fill_strategy());
  
  // check if each edge is present in at least one clique
  typedef undirected_edge<int> edge_type;
  foreach(edge_type e, g.edges()) {
    bool found = false;
    foreach(const node_set& clique, cliques) {
      if (clique.count(e.source()) && clique.count(e.target())) {
        found = true;
        break;
      }
    }
    BOOST_CHECK(found);
  }
}

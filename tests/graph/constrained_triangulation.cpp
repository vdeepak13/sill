#define BOOST_TEST_MODULE constrained_triangulation
#include <boost/test/unit_test.hpp>

#include <sill/graph/constrained_elim_strategy.hpp>
#include <sill/graph/grid_graph.hpp>
#include <sill/graph/min_degree_strategy.hpp>
#include <sill/graph/triangulation.hpp>
#include <sill/graph/undirected_graph.hpp>
#include <sill/model/junction_tree.hpp>

using namespace sill;

//! A function object that extracts the elimination priority from a graph
//! If needed, we could make this functor specific for graph_type below
struct elim_priority_functor {
  typedef size_t result_type;
  template <typename Graph>
  size_t operator()(typename Graph::vertex v, const Graph& graph) {
    return graph[v];
  }
};

BOOST_AUTO_TEST_CASE(test_triangulation) {
  // The graph type.  Each vertex is annotated with the elimination priority
  typedef undirected_graph<size_t, size_t> graph_type;

  // The clique type
  typedef std::set<size_t> node_set;

  // Build an m x 2 lattice.
  size_t m = 5;
  graph_type lattice;

  // Add the vertices and prioritize their elimination so that
  // vertices in the second column (ids 6-10) have a lower
  // elimination priority than vertices in the first column (ids 1-5).
  arma::umat v = make_grid_graph(m, 2, lattice);
  for(size_t i = 0; i < m; i++) {
    for(size_t j = 0; j < 2; j++) {
      lattice[v(i,j)] = j;
    }
  }

  // Create a constrained elimination strategy (using min-degree as
  // the secondary strategy).
  constrained_elim_strategy<elim_priority_functor, min_degree_strategy> s;

  // Create a junction tree using this elimination strategy.
  // (If we imagine vertices 1-5 as being discrete and vertices 6-10 as
  // continuous random variables, then this creates a strongly-rooted
  // junction tree.)
  junction_tree<size_t> jt(lattice, s);

  // Vertices
  // 1: ({5 9 10}  0)
  // 2: ({1 6 7}  0)
  // 3: ({1 2 7 8}  0)
  // 4: ({4 5 8 9}  0)
  // 5: ({1 2 3 4 5 8}  0)
  
  // Edges
  // 4 -- 5
  // 3 -- 5
  // 2 -- 3
  // 1 -- 4

  boost::array<size_t, 3> clique1 = {{5, 9, 10}};
  boost::array<size_t, 3> clique2 = {{1, 6, 7}};
  boost::array<size_t, 4> clique3 = {{1, 2, 7, 8}};
  boost::array<size_t, 4> clique4 = {{4, 5, 8, 9}};
  boost::array<size_t, 6> clique5 = {{1, 2, 3, 4, 5, 8}};

  std::vector<node_set> cliques;
  cliques.push_back(node_set(clique1.begin(), clique1.end()));
  cliques.push_back(node_set(clique2.begin(), clique2.end()));
  cliques.push_back(node_set(clique3.begin(), clique3.end()));
  cliques.push_back(node_set(clique4.begin(), clique4.end()));
  cliques.push_back(node_set(clique5.begin(), clique5.end()));
  
  junction_tree<size_t> jt2(cliques);
  
  BOOST_CHECK(jt == jt2);
}

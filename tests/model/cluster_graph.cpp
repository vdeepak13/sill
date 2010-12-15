#include <iostream>
#include <functional>

#include <prl/model/cluster_graph.hpp>
#include <prl/base/universe.hpp>

/**
 * \file cluster_graph.cpp Cluster Graph test
 */
int main() {

  using namespace prl;
  using namespace std;

  /**
  // Create a set of cliques.
  universe u;
  std::vector<variable_h> vars;
  for (int i = 0; i < 5; i++)
    vars.push_back(u.new_finite_variable(2));
  set<domain> cliques;
  cliques.insert(make_domain(vars[0], vars[1]));
  cliques.insert(make_domain(vars[0], vars[2], vars[3]));
  cliques.insert(make_domain(vars[3], vars[4], vars[5]));
  cliques.insert(make_domain(vars[4]));

  // Create a cluster graph (graph without factors).
  cluster_graph<> cg(cliques);

  std::cout << "Created cluster graph:\n" << cg << "\nChecking validity...";

  cg.check_validity();

  std::cout << "OK" << std::endl;
  */

  universe u;
  finite_var_vector vars = u.new_finite_variables(6, 2);
  
  cluster_graph<finite_variable*> cg;
  
  cg.add_cluster(1, make_domain(vars[0], vars[1]));
  cg.add_cluster(2, make_domain(vars[1], vars[2], vars[3]));
  cg.add_cluster(3, make_domain(vars[2], vars[3], vars[4]));
  cg.add_cluster(4, make_domain(vars[3], vars[5]));
  cg.add_edge(1, 2);
  cg.add_edge(2, 3);
  cg.add_edge(2, 4);
  
  cout << cg << endl;
  cout << "Connected: " << cg.connected() << endl;
  cout << "RIP: " << cg.running_intersection() << endl;

  cout << "Subgraph:" << endl;
  cluster_graph<finite_variable*> subgraph = cg.subgraph(1, 5);
  cout << subgraph << endl;

  cg.remove_edge(2, 4);
  cout << cg << endl;
  cout << "Connected: " << cg.connected() << endl;
  cout << "RIP: " << cg.running_intersection() << endl;

  cout << "Subgraph:" << endl;
  cout << cg.subgraph(1, 5) << endl;

  return EXIT_SUCCESS;
}

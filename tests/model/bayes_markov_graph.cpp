#include <iostream>

#include <sill/base/universe.hpp>
#include <sill/model/bayesian_graph.hpp>
#include <sill/model/markov_graph.hpp>

/**
 * \file bayes_markov_graph.cpp Bayes net graph and Markov graph test
 */
int main() {

  using namespace sill;
  using namespace std;

  universe u;
  finite_var_vector varvec = u.new_finite_variables(5, 2);
  finite_domain vars(varvec.begin(), varvec.end());
  bayesian_graph<finite_variable*> bg(vars);
  /* Create graph:
   * 0 --> 4
   * 1 --> 2 3
   * 2 --> 3
   * 3 --> 4
   * 4 --> 
   */
  bg.add_edge(varvec[0], varvec[4]);
  bg.add_edge(varvec[1], varvec[2]);
  bg.add_edge(varvec[1], varvec[3]);
  bg.add_edge(varvec[2], varvec[3]);
  bg.add_edge(varvec[3], varvec[4]);

  cout << "Bayes net graph with 5 nodes:\n" << bg << endl;

  markov_graph<finite_variable*> mg = bayes2markov_graph(bg);

  cout << "Converted Bayes net graph to Markov graph:\n" << mg << endl;

}

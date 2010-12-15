#include <fstream>
#include <iostream>

#include <boost/lexical_cast.hpp>

#include <prl/factor/table_factor.hpp>
#include <prl/model/markov_network.hpp>
#include <prl/model/io.hpp>
#include <prl/learning/ipf.hpp>

#include <prl/macros_def.hpp>

int main(int argc, char** argv) {
  using namespace prl;
  using namespace std;

  // The number of iterations
  size_t niters = (argc > 1) ? boost::lexical_cast<size_t>(argv[1]): 10;

  // Type definitions
  typedef pairwise_markov_network< tablef > mn_type;
  typedef mn_type::edge edge;
  universe u;
 
  // Load a pairwise Markov network from the file
  mn_type mn;
  ifstream in("../../../../tests/data/6x6.net");
  read_model(in, mn, u);
  shafer_shenoy< tablef > ss(mn);
  ss.calibrate();
  ss.normalize();

  // Compute the marginals for all edges
  std::vector< tablef > marginals;
  foreach(edge e, mn.edges())
    marginals.push_back(ss.belief(mn.nodes(e)));
  cout << "Tree width: " << ss.tree_width() << endl;
  cout << "Marginals: " << marginals << endl;

  // Compute the maximum-entropy model using IPF
  jt_ipf< tablef > ipf(marginals);
  double error = ipf.iterate(niters);
  cout << "Error after " << niters << " iterations: " << error << endl;
}

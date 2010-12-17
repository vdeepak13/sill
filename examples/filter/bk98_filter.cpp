#include "highway_dbn.hpp"

#include <boost/lexical_cast.hpp>

#include <prl/factor/table_factor.hpp>
#include <prl/model/dynamic_bayesian_network.hpp>
#include <prl/model/junction_tree.hpp>
#include <prl/inference/bk98_filter.hpp>

int main(int argc, char** argv) {
  using namespace prl;
  using namespace std;
  size_t nsegs  = argc > 1 ? boost::lexical_cast<size_t>(argv[1]) : 3;
  size_t nsteps = argc > 2 ? boost::lexical_cast<size_t>(argv[2]) : 2;
  size_t csize  = argc > 3 ? boost::lexical_cast<size_t>(argv[3]) : 1;
  bool print_joint = argc > 4 ? boost::lexical_cast<bool>(argv[4]) : true;
  assert(csize>0 && csize <= nsegs);

  // create the DBN
  std::vector<finite_timed_process*> procs;
  dynamic_bayesian_network< table_factor > dbn;
  highway_dbn(nsegs, dbn, procs);
  dbn.check_valid();
  cout << dbn << endl;
  
  // create the approximation structure (pairs of variables)
  finite_var_vector vars_t = variables(procs, current_step);
  std::vector<finite_domain> cliques;
  for(size_t i = 0; i <= nsegs - csize; i++)
    cliques.push_back(finite_domain(&vars_t[i], &vars_t[i]+csize));
  junction_tree<finite_variable*> jt(cliques);
  cout << "Approximation structure: " << jt << endl;

  bk98_filter< table_factor > filter(dbn, jt, false);
  
  cout << "t=0: " << filter.belief() << endl;
  for(size_t t = 1; t <= nsteps; t++) {
    filter.advance();
    cout << "t=" << t << ": ";
    if (print_joint) {
      cout << filter.belief() << endl;
    } else {
      for(size_t i = 0; i < nsegs; i++)
        cout << filter.belief(vars_t[i]) << endl;
    }
  }
  
  // TODO:
  // Test the result of flat_filter against variable elimination 
  // on the unrolled network
//   bayesian_network< table_factor > bn = dbn.unroll(nsteps);
//   cout << bn << endl;
}

#include "highway_dbn.hpp"

#include <boost/lexical_cast.hpp>

#include <prl/factor/table_factor.hpp>
#include <prl/model/dynamic_bayesian_network.hpp>
#include <prl/inference/flat_filter.hpp>

int main(int argc, char** argv) {
  using namespace prl;
  using namespace std;
  size_t nsegs  = argc > 1 ? boost::lexical_cast<size_t>(argv[1]) : 3;
  size_t nsteps = argc > 2 ? boost::lexical_cast<size_t>(argv[2]) : 2;
  bool print_joint = argc > 3 ? boost::lexical_cast<bool>(argv[3]) : true;

  dynamic_bayesian_network< tablef > dbn;
  std::vector<finite_timed_process*> procs;
  highway_dbn(nsegs, dbn, procs);
  dbn.check_valid();
  cout << dbn << endl;

  finite_var_vector vars_t = variables(procs, current_step);
  flat_filter< tablef > filter(dbn);
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
//   bayesian_network< tablef > bn = dbn.unroll(nsteps);
//   cout << bn << endl;
}

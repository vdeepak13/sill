#include <iostream>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <map>
#include <sill/parsers/detect_file_format.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/factor/canonical_table.hpp>
#include <sill/parsers/alchemy.hpp>
#include <sill/macros_def.hpp>

using namespace std;
using namespace sill;

typedef factor_graph_model<canonical_table> factor_graph_model_type;
typedef factor_graph_model_type::factor_type factor_type;

const double SPLITPROB = 0.3;
int main(int argc, const char* argv[]) {
  assert(argc == 3);
  universe u;
  factor_graph_model_type fg;
  string filename(argv[1]);
 
  cout << "Filename: " << filename << endl
       << "=========================================================="
       << endl;
  
  // Do the actual parsing.
  parse_alchemy(u, fg, filename);

  // read the input data
  if (parse_factor_graph(filename, u, fg) == false) {
    std::cout << "Unable to read factor graph\n";
  }
  
  cout << fg.arguments().size() << " variables" << endl;

  // now we need to replicate the factor graph how do we do this?
  // get all the variables and randomly decide whether to split or merge
  
  finite_var_map vmap;
  foreach(finite_variable* v, fg.arguments()) {
    if (double(rand())/RAND_MAX < SPLITPROB) {
      // if we are splitting, create a new finite variable of the same size
      vmap[v] = u.new_finite_variable(std::string(*v)+"p",v->size());
    }
    else {
      vmap[v] = v;
    }
  }
  
  std::vector<canonical_table> new_factors;
  foreach(const canonical_table &f, fg.factors()) {
    // check if there are any changes to this factor
    bool changes = false;
    foreach(finite_variable* v, f.arguments()) {
      if (vmap[v] != v) {
        changes = true;
        break;
      }
    }
    if (changes) {
      canonical_table newf = f;
      newf.subst_args(vmap);
      new_factors.push_back(newf);
    }
  }
  for (size_t i = 0; i < new_factors.size(); ++i) {
    fg.add_factor(new_factors[i]);
  }
  // write out
  ofstream fout;
  fout.open(argv[2]);
  print_alchemy(fg, fout);
  fout.close();
  return EXIT_SUCCESS;
}

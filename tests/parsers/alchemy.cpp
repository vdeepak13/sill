
#include <iostream>
#include <stdlib.h>
#include <string>
#include <fstream>

#include <prl/factor/table_factor.hpp>
#include <prl/factor/log_table_factor.hpp>
#include <prl/parsers/alchemy.hpp>


using namespace std;
using namespace prl;

typedef factor_graph_model<log_table_factor> factor_graph_model_type;
typedef factor_graph_model_type::factor_type factor_type;


int main(int argc, const char* argv[]) {
  cout << "Testing the Alchemy parser." << endl;

  assert(argc == 2);
  universe u;
  factor_graph_model_type fg;
  string filename(argv[1]);
 
  cout << "Filename: " << filename << endl
       << "=========================================================="
       << endl;
  
  // Do the actual parsing.
  parse_alchemy(u, fg, filename);

  ofstream fout("bla.txt");
  print_alchemy(fg, fout);
  fout.close();

  //  cout << fg.arguments() << endl;


  return EXIT_SUCCESS;
}

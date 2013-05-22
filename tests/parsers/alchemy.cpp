
#include <iostream>
#include <stdlib.h>
#include <string>
#include <fstream>

#include <sill/factor/table_factor.hpp>
#include <sill/factor/log_table_factor.hpp>
#include <sill/parsers/alchemy.hpp>


using namespace std;
using namespace sill;

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

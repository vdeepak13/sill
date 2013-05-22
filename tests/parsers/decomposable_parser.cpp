
#include <iostream>
#include <stdlib.h>
#include <string>

#include <sill/factor/table_factor.hpp>
#include <sill/parsers/decomposable_parser.hpp>

using namespace std;
using namespace sill;

int main(int argc, const char* argv[]) {
  cout << "Testing the decomposable parser." << endl;

  assert(argc == 2);
  decomposable<table_factor> fg;
  string filename(argv[1]);
  universe u;

  cout << "Filename: " << filename << endl
       << "=========================================================="
       << endl;
  // Do the actual parsing.
  parse_decomposable(u, fg, filename);

  cout << fg.arguments() << endl;


  return EXIT_SUCCESS;
}

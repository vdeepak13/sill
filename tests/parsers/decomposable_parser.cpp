
#include <iostream>
#include <stdlib.h>
#include <string>

#include <prl/factor/table_factor.hpp>
#include <prl/parsers/decomposable_parser.hpp>

using namespace std;
using namespace prl;

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

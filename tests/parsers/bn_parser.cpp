
#include <iostream>
#include <stdlib.h>
#include <string>

#include <sill/factor/table_factor.hpp>
#include <sill/parsers/bn_parser.hpp>

using namespace std;
using namespace sill;

int main(int argc, const char* argv[]) {
  cout << "Testing the Bayes net parser." << endl;

  assert(argc == 2);
  bayesian_network<table_factor> bn;
  string filename(argv[1]);
  universe u;

  cout << "Filename: " << filename << endl
       << "=========================================================="
       << endl;
  // Do the actual parsing.
  parse_bn(u, bn, filename);

  bn.check();

  cout << bn.arguments() << endl;

  return EXIT_SUCCESS;
}

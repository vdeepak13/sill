
#include <iostream>
#include <stdlib.h>
#include <string>

#include <prl/factor/table_factor.hpp>
#include <prl/parsers/bn_parser.hpp>

using namespace std;
using namespace prl;

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

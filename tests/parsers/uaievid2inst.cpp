#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <src/prl/parsers/string_functions.hpp>
using namespace prl;


int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout << "Usage: uaievid2inst [input uai evidence] [output ace inst file]" << std::endl;
    return -1;
  }

  std::ifstream fin(argv[1]);
  std::ofstream fout(argv[2]);
  size_t numevid;
  fin >> numevid;
  fout << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl;
  fout << "<instantiation>" << std::endl;
  for (size_t i = 0;i < numevid; ++i) {
    size_t varid, assg;
    fin >> varid >> assg;
    // <inst id = "__[varid]" value="[assg]"/>
    fout << "<inst id=\"" << "__" << varid << "\" value = \"" << assg << "\"/>" << std::endl;
  }
  fout << "</instantiation>";
}


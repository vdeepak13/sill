#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <boost/lexical_cast.hpp>

#include <src/prl/parsers/string_functions.hpp>
using namespace prl;


std::string build_state_list(size_t arity) {
  std::string statelist;
  for (size_t i = 0;i < arity; ++i) {
    statelist = statelist + "\"" + boost::lexical_cast<std::string>(i) + "\" ";
  }
  return statelist;
}

std::string build_potential_name(std::vector<size_t> &domain,
                                 std::vector<std::string> &varnames) {
  std::string potname;
  potname = "(" + varnames[domain[domain.size() - 1]] + " ";

  if (domain.size() > 1) {
    potname = potname + "| ";
  }
  
  for (size_t i = 0;i < domain.size() - 1 ; ++i) {
    potname = potname + varnames[domain[i]] + " ";
  }
  potname = potname + ")";
  return potname;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout << "Usage: uai2hugin [input uai file] [output hugin file]" << std::endl;
    std::cout << "  UAI file must be a Bayes net" << std::endl;
  }

  // we will be reading and performing the conversion at the same time
  // since the file structures are extremely similar.
  std::ifstream fin(argv[1]);
  std::ofstream fout(argv[2]);

  std::string header;
  fin >> header;
  header = trim(header);

  if (header != "BAYES") {
    std::cout << "Input file is not a UAI Bayes Net" << std::endl;
    return -1;
  }

  size_t numvars;
  fin >> numvars;

  std::vector<std::string> varnames;
  std::vector<size_t> arities;
  for (size_t i = 0;i < numvars; ++i) {
    // name the variable
    std::string varname = boost::lexical_cast<std::string>(i);
    varname = "__" + varname;
    size_t arity;
    fin >> arity;
    
    varnames.push_back(varname);
    arities.push_back(arity);
    // write the node
    fout << "node " << varname <<                     std::endl;
    fout << "{" <<                                    std::endl;
    fout <<     "\tstates = (" << build_state_list(arity) << ");" <<  std::endl;
    fout << "}" <<                                    std::endl;
  }


    // Second section is
    // # factors
    // [domain of factor 1]
    // [domain of factor 2]
    // ....

    size_t numfactors;
    fin >> numfactors;
    std::vector<std::vector<size_t> > domains;
    // read the domains
    for (size_t i = 0; i < numfactors; ++i) {
      // read the number of dimensions
      size_t numdim;
      fin >> numdim;
      std::vector<size_t> domain;
      // read each variable id and construct an empty factor
      for (size_t dim = 0;dim < numdim;++dim) {
        size_t varid;
        fin >> varid;
        domain.push_back(varid);
      }
      domains.push_back(domain);
    }

    // Third section is
    // [#vals in factor 1]
    //  [list of values in factor 1]
    // [#vals in factor 2]
    //  [list of values in factor 2]

    for (size_t i = 0; i < numfactors; ++i) {
      size_t numelem;
      fin >> numelem;

      // write the potential
      // now conveniently, the data is in the same order to write out in the hugin format
      fout << "potential " << build_potential_name(domains[i], varnames) << std::endl;      
      fout << "{" << std::endl;
      fout <<     "\tdata = (";

      size_t curvar = *(domains[i].end() - 1);
      
      for (size_t j = 0;j < numelem; ++j) {
        if (j % arities[curvar] == 0) {
          fout << "\t( ";
        }
        double val;
        fin >> val;
        fout << val << " ";
        if ((j+1) % arities[curvar] == 0) {
          fout << ")\n";
        }
      }
      fout << "\t);" << std::endl;
      fout << "}" <<    std::endl;
    }
    fin.close();
    fout.close();
}

#ifndef SILL_UAI_PARSER_HPP
#define SILL_UAI_PARSER_HPP

#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include <boost/lexical_cast.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/factor/log_table_factor.hpp>
#include <sill/model/factor_graph_model.hpp>

#include <sill/parsers/string_functions.hpp>

#include <sill/macros_def.hpp>
namespace sill {
  // parses the formar described in http://graphmod.ics.uci.edu/uai08/FileFormat
  /**
   * This function parses the file format described in
   * http://graphmod.ics.uci.edu/uai08/FileFormat
   */
  template <typename F>
  bool parse_uai(universe& u,
                factor_graph_model<F>& fg,
                const std::string& filename) {
    bool SATwarningprinted = false;
    std::ifstream fin(filename.c_str());
    std::string preamble;
    std::getline(fin, preamble);
    preamble = trim(preamble);

    // read the preamble
    if (preamble == "MARKOV" || preamble == "BAYES") {
      std::cout << "filetype is " << preamble << std::endl;
    }
    else {
      std::cout << "UAI Format Preamble not found!" << std::endl;
      return false;
    }

    // now, since we are constructing a factor graph, we don't
    // really care about the format, Markov or Bayes

    // First section is
    // # variables
    // [arity of all variables in sequence]
    
    size_t numvars;
    fin >> numvars;
    std::vector<finite_variable*> variables;
    variables.resize(numvars);
    // create the variables
    for (size_t i = 0;i < numvars; ++i) {
      // name the variable
      std::string varname = boost::lexical_cast<std::string>(i);
      size_t arity;
      fin >> arity;
      variables[i] = u.new_finite_variable(varname, arity);
    }


    // Second section is
    // # factors
    // [domain of factor 1]
    // [domain of factor 2]
    // ....

    size_t numfactors;
    fin >> numfactors;
    std::vector<finite_var_vector> factordomains;
    factordomains.resize(numfactors);
    // read the domains
    for (size_t i = 0; i < numfactors; ++i) {
      // read the number of dimensions
      size_t numdim;
      fin >> numdim;
      finite_var_vector d;
      // read each variable id and construct an empty factor
      for (size_t dim = 0;dim < numdim;++dim) {
        size_t varid;
        fin >> varid;
        d.push_back(variables[varid]);
      }
      // reverse d.
      // this will make it match up with the order of the function entries
      std::reverse(d.begin(), d.end());
      factordomains[i] = d;
    }

    // Third section is
    // [#vals in factor 1]
    //  [list of values in factor 1]
    // [#vals in factor 2]
    //  [list of values in factor 2]

    for (size_t i = 0; i < numfactors; ++i) {
      // read the number of dimensions
      size_t numelem;
      fin >> numelem;
      std::vector<typename F::result_type> elems;
      elems.resize(numelem);
      for (size_t j = 0;j < numelem; ++j) {
        double val;
        fin >> val;
        if (val == 0) {
          val = 1E-50;
          if (SATwarningprinted == false) {
            std::cout << "Warning: Factors have 0 values."
                         "These values will be stored as 1E-50" << std::endl;
            SATwarningprinted = true;
          }
        }
        elems[j] = val;
      }
      fg.add_factor(F(factordomains[i], elems));
    }
    fin.close();
    return true;
  } // end of parse_uai method

  finite_assignment parse_uai_evidence(universe& u,
                                   const std::string& filename) {
    finite_assignment asg;
    std::ifstream fin(filename.c_str());
    
    // read the number of assigned variables
    size_t numev = 0;
    fin >> numev;
    if (fin.fail()) {
      std::cout << "Unable to read evidence " << filename << std::endl;
      return asg;
    }
    
    for (size_t i = 0;i < numev; ++i) {
      // read the variable id and the value
      size_t varid, val;
      if (fin.fail()) return asg; 
      fin >>varid>> val;
      // the variable name is just the varid as a string
      std::string varname = boost::lexical_cast<std::string>(varid);
      // find the variable in the universe
      variable* v = u.var_from_name(varname);
      assert(v->get_variable_type() == variable::FINITE_VARIABLE);
      asg[dynamic_cast<finite_variable*>(v)] = val;
    }
    return asg;
  }
  
  template <typename F>
  void write_uai_beliefs(universe &u,
                         std::map<finite_variable*, F> beliefs,
                         std::string& output_filename) {
    // Create an ouptut filestream
    std::ofstream fout(output_filename.c_str());
    assert(fout.good());
    size_t numvars = beliefs.size();
    fout << "m " << numvars << " ";
    
    for(size_t i = 0; i < numvars; ++i) {
      std::string varname = boost::lexical_cast<std::string>(i);
      // find the variable in the universe
      variable* v = u.var_from_name(varname);
      assert(v->get_variable_type() == variable::FINITE_VARIABLE);
      
      finite_variable* f = dynamic_cast<finite_variable*>(v);
      fout << beliefs[f].size() << " ";
      for (size_t i = 0;i < beliefs[f].size(); ++i) {
        fout << (double)(beliefs[f](i)) << " ";
      }
      fout << std::endl;
    }
    
    fout.flush();
    fout.close();
  }

} // End of namespace
#include <sill/macros_undef.hpp>


#endif // SILL_BN_PARSER_HPP

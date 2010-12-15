#ifndef PRL_DETECT_FILE_FORMAT_HPP
#define PRL_DETECT_FILE_FORMAT_HPP

#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include <prl/factor/table_factor.hpp>
#include <prl/model/factor_graph_model.hpp>

#include <prl/parsers/protein.hpp>
#include <prl/parsers/uai_parser.hpp>
#include <prl/parsers/alchemy.hpp>
#include <prl/parsers/bif_parser.hpp>
#include <prl/macros_def.hpp>
namespace prl {

  enum filetypes_enum {
    FILETYPE_UAI,     // UAI Inference Challenge 2008 file format as 
                      // specified in http://graphmod.ics.uci.edu/uai08/

    FILETYPE_PROTEIN, // Internal Protein Pairwise MRF file format

    FILETYPE_BIF,     // Bayesian Interchange format as specified in 
                      // http://www.cs.cmu.edu/~fgcozman/Research/InterchangeFormat/

    FILETYPE_ALCHEMY, // Alchemy Factor Graph format. 
                      // At the moment I cannot differentiate between lifted
                      // and non-lifted

    FILETYPE_OTHER    // Any other format. Current parser defaults this to
                      // the file created by the PRL serializer
  };


  /**
    Uses some heuristics to detect the filetype
  */
  bool detect_file_type(std::string filename, filetypes_enum &filetype) {
    std::ifstream fin(filename.c_str());
    std::string line;
    // loop till I find a non-empty line
    while(line.length() == 0) {
      std::getline(fin, line);
      if (fin.fail()) break;
      std::string trimedline = trim(line);
      if (trimedline == "MARKOV" || trimedline == "BAYES") {
        filetype = FILETYPE_UAI;
        return true;
      }
      else if(trimedline == "variables:") {
        filetype = FILETYPE_ALCHEMY;
        return true;
      }
      else if(trimedline[trimedline.length() - 1] == '{' ||
              trimedline.substr(0,7) == "network") {
        filetype = FILETYPE_BIF;
        return true;
      }
      else {
        // it is unfortunately difficult to differentiate between
        // FILETYPE_PROTEIN and other binary formats
        // just try to load it
        universe unused;
        factor_graph_model<table_factor> fg;
        if (parse_protein(unused,fg,filename)) filetype = FILETYPE_PROTEIN;
        else filetype = FILETYPE_OTHER;
        return true;
      }
    }
    return false;
  }



  /**
  Uses the file extension to identify the different file types
  */
  template <typename FactorModel>
  bool parse_factor_graph(std::string input_filename, universe &u,
                          FactorModel &fg) {
    filetypes_enum filetype;
    if (detect_file_type(input_filename, filetype) == false) return false;
    if (filetype == FILETYPE_OTHER) {
      // binary serialized input
      std::ifstream fin;
      fin.open(input_filename.c_str());
      iarchive arc(fin);
      arc >> u;
      arc.attach_universe(&u);
      arc >> fg;
      fin.close();
      return true;
    }
    else if (filetype == FILETYPE_ALCHEMY) {
      // alchemy file
      return parse_alchemy(u, fg, input_filename);
    }
    else if (filetype == FILETYPE_PROTEIN) {
      // alchemy file
      return parse_protein(u, fg, input_filename);
    }
    else if (filetype == FILETYPE_UAI) {
      // alchemy file
      return parse_uai(u, fg, input_filename);
    }
    else if (filetype == FILETYPE_BIF) {
      // BIF file
      std::vector<finite_variable*> unused_elim;
      std::set<finite_variable*> unused_leaves;
      return parse_bif(u, fg,
                  unused_elim, unused_leaves,
                  input_filename);
    }
    else {
      std::cerr << "Unrecognized file format" << std::endl;
      return false;
    }
  }
} // End of namespace
#include <prl/macros_undef.hpp>


#endif // PRL_BN_PARSER_HPP

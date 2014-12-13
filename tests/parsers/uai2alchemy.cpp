
/**
 * This program generates information about a model in one of the
 * parsable formats.  It prints the number of variables, factors, and
 * edges.  In addition it generates additional statistics if an output
 * file is provided.
 */



// C Includes 
#include <cstdlib>
#include <cmath>
// STL Includes
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <limits>

// Boost includes
#include <boost/program_options.hpp>

// SILL Includes
#include <sill/model/factor_graph_model.hpp>
#include <sill/factor/canonical_table.hpp>

#include <sill/parsers/detect_file_format.hpp>



#include <sill/macros_def.hpp>


// Rename the program options namespace
namespace po = boost::program_options;

using namespace std;
using namespace sill;

typedef factor_graph_model<canonical_table>   factor_graph_type;
typedef factor_graph_type::factor_type         factor_type;
typedef factor_graph_type::vertex_type         vertex_type;
typedef factor_graph_type::variable_type       variable_type;



// Run the engine
int main(int argc, char* argv[]) {
  string uai_filename = "Not Provided";
  string uai_evid_filename = "Not Provided";
  string alchemy_filename = "Not Provided";

  // Parse the input
  po::options_description desc("Allowed Options");
  desc.add_options()
    ("uaimodel", po::value<string>(&uai_filename), 
     "UAI Model")
    ("uaievid", po::value<string>(&uai_evid_filename), 
     "UAI Model")
    ("alchemy", po::value<string>(&alchemy_filename), 
     "Output alchemy factor graph");
    
  po::positional_options_description pos_opts;
  pos_opts.add("uaimodel",1).add("alchemy", 1);
  po::variables_map vm;
  store(po::command_line_parser(argc, argv)
        .options(desc).positional(pos_opts).run(), vm);
  notify(vm);

  
  if( vm.count("help") || !vm.count("uaimodel") ||
      !vm.count("alchemy")  ) {
    cout << "Usage: " << argv[0] << " factorgraph.file " 
         << "[output_filename_base]" << endl
         << "This supports a wide variety of factor graph formats." << endl;
    cout << desc;
    return EXIT_FAILURE;
  }


  cout << "==========================================================" << endl
       << "Program Settings: " << endl
       << "UAI Model:    " << uai_filename << endl
       << "UAI Evidence: " << uai_evid_filename << endl
       << "Alchemy:      " << alchemy_filename << endl
       << "==========================================================" << endl;

  universe universe;
  factor_graph_type fg;

  // Do the actual parsing.
  bool success = parse_factor_graph(uai_filename, universe, fg);
  if(!success) {
    cout << "Unable to parse file!!!!" << endl;
    return EXIT_FAILURE;
  }
  assert(success);

  cout << "Simplifying Factor graph. ";
  fg.simplify();
  fg.normalize();
  cout << "Finished." << endl;


  if(vm.count("uaievid")) {
    cout << "Incorporating Evidence. ";
    finite_assignment evidence =
      parse_uai_evidence(universe, uai_evid_filename);
    fg.integrate_evidence(evidence);
    fg.simplify_stable();
    fg.normalize();
  }
  


  {
    cout << "Saving Alchemy file:";
    stringstream strm;
    strm << alchemy_filename << ".alchemy";
    ofstream fout(strm.str().c_str());
    assert(fout.good());
    print_alchemy(fg, fout);
    fout.close();
  }
  

  
  cout << "Computing number of edges. ";
  size_t edges = 0;
  size_t totalwork = 0;
  foreach(const vertex_type& v, fg.vertices()) {
    edges += fg.num_neighbors(v);
    totalwork += fg.work_per_update(v);
  }
  edges = edges/2;
  cout << " Done." << endl;
  
  cout << "Computing Maximum variable size. ";
  size_t maxarity = 0;
  foreach(const variable_type* v, fg.arguments()) {
    maxarity = max(maxarity, v->size());
  }
  cout << "Done" << endl;


  cout << "Computing max ratio." << endl;
  double max_ratio = 0;
  foreach(const factor_type& factor, fg.factors()) {
    const logarithmic<double> max = factor.maximum();
    const logarithmic<double> min = factor.minimum();
    const double ratio = log(max) - log(min);
    max_ratio = std::max(ratio, max_ratio);
  }
  cout << "Done" << std::endl;

  
  // Print statistics about the model
  cout << "Statistics: " << endl
       << "  - Variables:   " << fg.arguments().size() << endl
       << "  - Factors:     " << fg.size() << endl
       << "  - Edges:       " << edges << endl
       << "  - Work :       " << totalwork << endl
       << "  - Max Ratio:   " << max_ratio << endl
       << "  - Avg Work per vertex:     " 
       << double(totalwork) / (fg.arguments().size() + fg.size()) 
       << endl;

  {
    cout << "Saving Alchemy stats file:";
    stringstream strm;
    strm << alchemy_filename << "_stats.txt";
    ofstream fout(strm.str().c_str());
    assert(fout.good());
    fout << "Statistics:      " << uai_filename << endl
         << "  - Variables:   " << fg.arguments().size() << endl
         << "  - Factors:     " << fg.size() << endl
         << "  - Edges:       " << edges << endl
         << "  - Work :       " << totalwork << endl
         << "  - Max Ratio:   " << max_ratio << endl
         << "  - Avg Work per vertex:     " 
         << double(totalwork) / (fg.arguments().size() + fg.size()) 
         << endl;
    
    fout.close();
  }
  


  
  return EXIT_SUCCESS;
} // End of main



#include <sill/macros_undef.hpp>
//End of file





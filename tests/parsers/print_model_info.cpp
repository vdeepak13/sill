/**
 * This program generates information about a model in one of the
 * parsable formats.  It prints the number of variables, factors, and
 * edges.  In addition it generates additional statistics if an output
 * file is provided.
 */



// C Includes 
#include <cstdlib>

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

// PRL Includes
#include <prl/model/factor_graph_model.hpp>
#include <prl/factor/log_table_factor.hpp>

#include <prl/parsers/detect_file_format.hpp>



#include <prl/macros_def.hpp>


// Rename the program options namespace
namespace po = boost::program_options;

using namespace std;
using namespace prl;

typedef factor_graph_model<log_table_factor>   factor_graph_type;
typedef factor_graph_type::factor_type         factor_type;
typedef factor_graph_type::vertex_type         vertex_type;
typedef factor_graph_type::variable_type       variable_type;



// Run the engine
int main(int argc, char* argv[]) {
  string input_filename = "Not Provided";
  string output_filename = "Not Provided";

  // Parse the input
  po::options_description desc("Allowed Options");
  desc.add_options()
    ("infn", po::value<string>(&input_filename), 
     "factor graph model")
    ("outfn", po::value<string>(&output_filename), 
     "file to write the beliefs");
    
  po::positional_options_description pos_opts;
  pos_opts.add("infn",1).add("outfn", 1);
  po::variables_map vm;
  store(po::command_line_parser(argc, argv)
        .options(desc).positional(pos_opts).run(), vm);
  notify(vm);

  
  if( vm.count("help") || !vm.count("infn")  ) {
    cout << "Usage: " << argv[0] << " factorgraph.file " 
         << "[output_filename_base]" << endl
         << "This supports a wide variety of factor graph formats." << endl;
    cout << desc;
    return EXIT_FAILURE;
  }


  cout << "==========================================================" << endl
       << "Program Settings: " << endl
       << "Input filename:  " << input_filename << endl
       << "Output filename: " << output_filename << endl
       << "==========================================================" << endl;

  universe universe;
  factor_graph_type fg;

  // Do the actual parsing.
  bool success = parse_factor_graph(input_filename, universe, fg);
  if(!success) {
    cout << "Unable to parse file!!!!" << endl;
    return EXIT_FAILURE;
  }
  assert(success);

  cout << "Simplifying Factor graph. ";
  fg.simplify();
  cout << "Finished." << endl;
  
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

  // Print statistics about the model
  cout << "Statistics: " << endl
       << "  - Variables:   " << fg.arguments().size() << endl
       << "  - Factors:     " << fg.size() << endl
       << "  - Edges:       " << edges << endl
       << "  - Work :     " << totalwork << endl        
       << "  - Avg Work per vertex:     " 
       << double(totalwork) / (fg.arguments().size() + fg.size()) 
       << endl;


  if(!vm.count("outfn")) {
    return EXIT_SUCCESS;    
  }
  
  cout << "Additional information requested." << endl;
  std::stringstream degfn;
  degfn << output_filename << "-degdist.csv";
  ofstream foutdeg(degfn.str().c_str());
  assert(foutdeg.good());
  foreach(const vertex_type& v, fg.vertices()) {
    foutdeg << fg.num_neighbors(v) << ", "
            << v.is_factor() << endl;
  }
  foutdeg.close();


  std::stringstream factorsizefn;
  factorsizefn << output_filename << "-factorsize.csv";
  ofstream foutfactorsize(factorsizefn.str().c_str());
  assert(foutfactorsize.good());
  foreach(const factor_type& f, fg.factors()) {
    foutfactorsize << f.size() << ", "
                   << f.arguments().size() << ", "
                   << static_cast<double>(f.maximum()) << ", "
                   << static_cast<double>(f.minimum()) <<  endl;
  }
  foutfactorsize.close();

  return EXIT_SUCCESS;
} // End of main



#include <prl/macros_undef.hpp>
//End of file





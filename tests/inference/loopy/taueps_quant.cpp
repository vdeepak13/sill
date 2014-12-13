// C Includes 
#include <stdlib.h>

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
#include <boost/random/mersenne_twister.hpp>


// SILL Includes
#include <sill/serialization/serialize.hpp>

#include <sill/model/factor_graph_model.hpp>
#include <sill/factor/canonical_table.hpp>

#include <sill/parallel/timer.hpp>

#include <sill/parsers/detect_file_format.hpp>
#include <sill/parsers/alchemy.hpp>

#include <sill/inference/loopy/blf_residual_splash_bp.hpp>


#include <sill/factor/util/norms.hpp>
#include <sill/macros_def.hpp>


// Rename the program options namespace
namespace po = boost::program_options;

using namespace std;
using namespace sill;

typedef factor_graph_model<canonical_table>   factor_graph_type;
typedef factor_graph_type::factor_type         factor_type;
typedef factor_graph_type::vertex_type         vertex_type;
typedef factor_graph_type::variable_type       variable_type;

// Define the various engine types
typedef blf_residual_splash_bp<factor_type> bsplash_engine_type;
typedef bsplash_engine_type::message_type      message_type;




// Generate a random number between 0 and max
size_t randnext(size_t max) {
  size_t number = rand() % max;
  assert(number < max);
  return number;
}





// Run the engine
int main(int argc, char* argv[]) {
  string input_filename;
  string output_filename;
  double bound;
  double damping;
  size_t splashv;
  size_t ntraces;
  size_t walklen;

  // Parse the input
  po::options_description desc("Allowed Options");
  desc.add_options()
    ("infn", po::value<string>(&input_filename), 
     "MLN Factor graph")
    ("outfn", po::value<string>(&output_filename), 
     "file to write the beliefs")
    ("help", "produce help message")
    ("bound", po::value<double>(&bound)->default_value(0.001), 
     "accuracy bound")
    ("splashv", po::value<size_t>(&splashv)->default_value(100), 
     "volume of splash")
    ("damping", po::value<double>(&damping)->default_value(0.5), 
     "level of damping")
    ("ntraces", po::value<size_t>(&ntraces)->default_value(50), 
     "number of traces")
    ("walklen", po::value<size_t>(&walklen)->default_value(500), 
     "length of the walk");
    
  po::positional_options_description pos_opts;
  pos_opts.add("infn",1).add("outfn",1);
  po::variables_map vm;
  store(po::command_line_parser(argc, argv)
        .options(desc).positional(pos_opts).run(), vm);
  notify(vm);

  
  if(vm.count("help") || !vm.count("infn") ||  (!vm.count("outfn") )) {
    cout << "Usage: " << argv[0] << " [options] infn outfn" << endl;
    cout << desc;
    return EXIT_FAILURE;
  }


  cout << "==========================================================" << endl
       << "Program Settings: " << endl
       << "Input filename:  " << input_filename << endl
       << "Output filename: " << output_filename << endl
       << "bound:           " << bound << endl
       << "splash volume:   " << splashv << endl
       << "damping:         " << damping << endl
       << "ntraces:         " << ntraces << endl
       << "walklen:         " << walklen << endl
       << "==========================================================" << endl;

  universe universe;
  factor_graph_type fg;

  // Do the actual parsing.
  bool success = parse_factor_graph(input_filename, universe, fg);
  assert(success);

  cout << "Finished parsing: " << fg.arguments().size() 
       << " variables and " << fg.size() << " factors."
       << endl;
    
  // Initialize the results
  std::map<vertex_type, factor_type> blfs;
  finite_assignment mapassg;
  double executiontime;
  
  bsplash_engine_type engine(&fg, splashv, bound, damping);
  cout << "Performing Inference:" << endl;
  timer ti; ti.start();
  // Run the engine
  bool converged = engine.run();
  // Record execution type
  executiontime = ti.current_time();
  // Record the results
  engine.map_assignment(mapassg);
  engine.belief(blfs);
    
  cout << "Finished!" << endl;
  cout << "Took " << executiontime << " seconds" << endl;

  if(!converged) {
    cout << "Failed to converge, terminated early!" << endl;
    return EXIT_FAILURE;
  }

  cout << "Starting random change experiment " << endl;
  const std::vector<vertex_type>& vertices = fg.vertices();

  cout << "Setting damping to 0.0" << endl;
  // Turn off damping
  engine.damping(0.0);

  cout << "Opening outfile" << endl;

  std::stringstream blffn;
  blffn << output_filename << ".blf";
  ofstream foutblf(blffn.str().c_str());
  assert(foutblf.good());

  std::stringstream msgfn;
  msgfn << output_filename << ".msg";
  ofstream foutmsg(msgfn.str().c_str());
  assert(foutmsg.good());
  
  // Backup the state of the old messages
  bsplash_engine_type engine_backup = engine;

  for(size_t t = 0; t < ntraces; ++t) {
    cout << "===================================================" << endl
         << "Starting trace: " << t << endl;
    
    cout << "Selecting origin" << endl;
    // get the source and destination of the first message
    vertex_type source = vertices[randnext(vertices.size())];
    std::vector<vertex_type> neighbors = fg.neighbors(source);
    vertex_type target = neighbors[randnext(neighbors.size())];


    // Zap the message
    const message_type& msg = engine.message(source, target);
    std::pair<double,double> residual = 
      engine.update_message(source, target, 
                            message_type(msg.arguments(), 1.0).normalize());
    
    // print some startup output
    cout << "Starting Message------------- " << endl
         << "Source   : " << source << endl
         << "Target   : " << target << endl
         << "Blf Residual : " << residual.first << endl
         << "Msg Residual : " << residual.second << endl;

    // the residual
    foutblf << residual.first << ", ";
    foutmsg << residual.second << ", ";
    
    
    // Take random walk passing modified message
    for(size_t i = 0; i < walklen; ++i) {
      source = target;
      neighbors = fg.neighbors(source);
      vertex_type target = neighbors[randnext(neighbors.size())];
      residual = engine.send_message(source, target);
      cout << "Residual (" << i << ") -> " 
           << residual.first << ", " << residual.second << endl;
      foutblf << residual.first;
      foutmsg << residual.second;
      // If not the last entry append a comma
      if(i < (walklen -1)) {
        foutblf << ", ";
        foutmsg << ", ";
      }
    }
  
    // add a newline
    foutblf << endl;
    foutmsg << endl;

    // Recover the old engine
    engine = engine_backup;

  } // End of for loop

  foutblf.close();
  foutmsg.close();


  return EXIT_SUCCESS;

} // End of main



#include <sill/macros_undef.hpp>
//End of file





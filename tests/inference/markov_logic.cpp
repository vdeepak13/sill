#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <map>
#include <limits>

#include <boost/program_options.hpp>
#include <sill/serialization/serialize.hpp>


#include <sill/model/factor_graph_model.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/factor/log_table_factor.hpp>

#include <sill/parallel/timer.hpp>
#include <sill/parsers/alchemy.hpp>

#include <sill/inference/blf_residual_splash_bp.hpp>
#include <sill/inference/residual_splash_bp.hpp>
#include <sill/inference/gibbs_engine.hpp>
#include <sill/inference/round_robin_bp.hpp>
#include <sill/inference/mk_propagation.hpp>

#include <sill/factor/norms.hpp>
#include <sill/macros_def.hpp>


// Rename the program options namespace
namespace po = boost::program_options;

using namespace std;
using namespace sill;


typedef factor_graph_model<log_table_factor>   factor_graph_type;
typedef factor_graph_type::factor_type   factor_type;
typedef factor_graph_type::vertex_type   vertex_type;
typedef factor_graph_type::variable_type variable_type;


// Define the various engine types
typedef residual_splash_bp<factor_type>     rsplash_engine_type;
typedef round_robin_bp<factor_type>         rr_engine_type;
typedef gibbs_engine<factor_type>           gibbs_engine_type;
typedef blf_residual_splash_bp<factor_type> bsplash_engine_type;




// Save the beliefs of the converged engine
template<typename EngineType>
void save_beliefs(factor_graph_type& fg,
                  EngineType& engine,
                  string& output_filename) {
  // Create an ouptut filestream
  ofstream fout(output_filename.c_str());
  assert(fout.good());
  fout.precision(10);
  foreach(variable_type* v, fg.arguments()) {
    fout<<"\"" << v->name() << "\", "  << (double)(engine.belief(v).v(1)) << endl;
  }
  fout.flush();
  fout.close();
} // save the beliefs


// Save the map assignments
void save_map_estimates(factor_graph_type& fg,
                        finite_assignment mapassg,
                        string& output_filename) {
  // Create an ouptut filestream
  ofstream fout(output_filename.c_str());
  assert(fout.good());
  foreach(variable_type* v, fg.arguments()) {
    if (mapassg[v]) fout << v->name() << std::endl;
  }
  fout.close();
} // end save map estimates


// Load the factor graph from a file.
void load_factor_graph(universe& universe,
                       factor_graph_type& fg,
                       string& input_filename) {
  // Do the actual parsing.
  // See support parsing in several formats
  if (input_filename.substr(input_filename.length()-4,4) == ".bin") {
    std::ifstream fin;
    fin.open(input_filename.c_str());
    iarchive arc(fin);
    arc >> universe;
    arc.attach_universe(&universe);
    arc >> fg;
    fin.close();
  } else {
    parse_alchemy(universe, fg, input_filename);
    fg.simplify_stable();
    fg.normalize();
  }
} // end of load_factor_graph



// Save statistics about the factor graph as well as a binary version
// for later processing
void save_statistics(universe& universe,
                     factor_graph_type& fg) {
  std::ofstream fout;
  
  fout.open("adjacency.txt");  
  fg.print_adjacency(fout);
  fout.close();

  fout.open("vertinfo.txt");
  fg.print_vertex_info(fout);
  fout.close();
  
  fout.open("factorgraph.bin");
  oarchive arc(fout);
  arc << universe;
  arc << fg;
  fout.flush();
  fout.close();  
} // end save_statistics





// Run the engine
int main(int argc, char* argv[]) {
  string engine_name;
  string input_filename;
  string output_filename;
  double bound;
  double damping;
  size_t splashv;
  bool adjonly;
  size_t adjprop;
  string truthdir;
  string belieffile;
  size_t tracelevel;

  // Parse the input
  po::options_description desc("Allowed Options");
  desc.add_options()
    ("engine", po::value<string>(&engine_name), 
     "Engine Type [roundrobin, rpslash, bsplash, gibbs] ")
    ("infn", po::value<string>(&input_filename), 
     "MLN Factor graph")
    ("outfn", po::value<string>(&output_filename), 
     "file to write the beliefs")
    ("help", "produce help message")
    ("bound", po::value<double>(&bound)->default_value(0.001), 
     "accuracy bound")
    ("splashv", po::value<size_t>(&splashv)->default_value(100), 
     "volume of splash")
    ("damping", po::value<double>(&damping)->default_value(0.7), 
     "volume of splash")
    ("truthdir", po::value<string>(&truthdir)->default_value(""),
     "directory of truth")
    ("belieffile", po::value<string>(&belieffile)->default_value(""),
     "file of true belief values")
    ("adjonly", "Output adjacency file only")
    ("adjprop", po::value<size_t>(&adjprop)->default_value(0),
     "number of MK propagations")
    ("tracelevel", po::value<size_t>(&tracelevel)->default_value(0),
    "1: output likelihood and energy each second. 2: output update trace");
    
  po::positional_options_description pos_opts;
  pos_opts.add("engine",1).add("infn",1).add("outfn",1);
  po::variables_map vm;
  store(po::command_line_parser(argc, argv)
        .options(desc).positional(pos_opts).run(), vm);
  notify(vm);

  
  if(vm.count("help") || 
     !vm.count("engine") || !vm.count("infn") ||  (!vm.count("outfn") )) {
    cout << "Usage: " << argv[0] << " [options] infn outfn" << endl;
    cout << desc;
    return EXIT_FAILURE;
  }
  adjonly = vm.count("adjonly") > 0;
  (void)adjonly; // to avoid warnings

  cout << "==========================================================" << endl
       << "Program Settings: " << endl
       << "Engine:          " << engine_name << endl
       << "Input filename:  " << input_filename << endl
       << "Output filename: " << output_filename << endl
       << "bound:           " << bound << endl
       << "splash volume:   " << splashv << endl
       << "damping:         " << damping << endl
       << "==========================================================" << endl;

  universe universe;
  factor_graph_type fg;

  // Do the actual parsing.
  load_factor_graph(universe, fg, input_filename);

  cout << "Finished parsing: " << fg.arguments().size() 
       << " variables and " << fg.size() << " factors."
       << endl;
  
  // Print the degree distribution
  fg.print_degree_distribution(); 

  // save some statistics
  // save_statistics(universe, fg);

  finite_assignment truth;
  std::map<finite_variable*, log_table_factor> truebeliefs;
  
  if (truthdir.length() > 0) {
    alchemy_parse_truthdir(universe,truth,truthdir,true);
    cout << "Parsed truth data" << endl;
  }

  if (belieffile.length() > 0) {
    alchemy_parse_belief_file(universe,truebeliefs,belieffile);
    cout << "Parsed truth data" << endl;
  }
  
  // Initialize the results
  std::map<vertex_type, factor_type> blfs;
  finite_assignment mapassg;
  double executiontime;
  
  if(engine_name.compare("roundrobin")) {
    cout << "Not currently supported!!!!!!!!" << endl;
    return EXIT_FAILURE;
    //rr_engine_type engine(&fg, splashv, bound, damping);
  } else if(engine_name.compare("rsplash")) {
    cout << "Not currently supported!!!!!!!!" << endl;
    return EXIT_FAILURE;
    // rsplash_engine_type engine(&fg, splashv, bound, damping);
  } else if(engine_name.compare("bsplash")) {
    bsplash_engine_type engine(&fg, splashv, bound, damping);
    cout << "Performing Inference:" << endl;
    timer ti; ti.start();
    // Run the engine
    engine.run();
    // Record execution type
    executiontime = ti.current_time();
    // Record the results
    engine.map_assignment(mapassg);
    engine.belief(blfs);
  } else if(engine_name.compare("gibbs")) { 
    cout << "Not currently supported!!!!!!!!" << endl;
    return EXIT_FAILURE;
    // cpu_time_convergence_measure convmeasure(bound, 10000000.0);
    // gibbs_engine_type engine(&fg, splashv, &convmeasure, damping);
  } else {
    cout << "Not a valid engine!!!!!!!!" << endl;
    return EXIT_FAILURE;
  }
    

  cout << "Finished!" << endl;
  cout << "Took " << executiontime << " seconds" << endl;
  return EXIT_SUCCESS;

} // End of main





//////// Undocumented Yucheng stuff

//   std::map<variable_type*, factor_type> *truebeliefptr = NULL;

//   double truth_error=0;
//   logarithmic<double> truth_logprob=0;
//   double kldiff = 0;
  
//   if (truthdir.length() > 0) {
//     truth_error=alchemy_compute_error(fg,engine,truth);
//     truth_logprob=alchemy_compute_truthlogprob(fg,engine,truth);
//   }
//   if (belieffile.length() > 0) {
//     factor_norm_1<log_table_factor> norm;
//     foreach(finite_variable* i, fg.arguments()) {
//       double d = norm(truebeliefs[i],engine.belief(i));
//       //double d = truebeliefs[i].relative_entropy(engine.belief(i));
//       kldiff+=d;
//     }
//   }

//   std::cout.precision(10);
//   //std::cout << "Energy: " << fg.bethe(blfs) << std::endl;
//   std::cout << "Log Likelihood: " << fg.log_likelihood(mapassg) << std::endl;
// #ifdef BLF_SPLASH
//   std::cout << "Splash Count: " << engine.splash_count()  << std::endl;
//   std::cout << "Update Count: " << engine.update_count() << std::endl;
// #endif
//   std::cout << "Execution Time: " << executiontime << std::endl;
//   if (truthdir.length() > 0) {
//     std::cout << "Truth Error: " << truth_error << std::endl;
//     std::cout << "Truth LogProb: " << truth_logprob<< std::endl;
//   }
//   if (belieffile.length() > 0) {
//     std::cout << "KL Divergence: " << kldiff<< std::endl;
//   }


//   fout.open("results.txt");
//   fout.precision(10);
//   //fout << "Energy: " << fg.bethe(blfs) << std::endl;
//   fout << "Log Likelihood: " << fg.log_likelihood(mapassg) << std::endl;
// #ifdef BLF_SPLASH
//   fout << "Splash Count: " << engine.splash_count() << std::endl;
//   fout << "Update Count: " << engine.update_count() << std::endl;
// #endif
//   fout << "Execution Time: " << executiontime << std::endl;
//   if (truthdir.length() > 0) {
//     fout << "Truth Error: " << truth_error << std::endl;
//     fout << "Truth LogProb: " << truth_logprob<< std::endl;
//   }
//   if (belieffile.length() > 0) {
//     fout << "KL Divergence: " << kldiff<< std::endl;
//   }

//   fout.close();


//   #ifdef BLF_SPLASH
//   fout.open("degcounts.txt");
//   foreach(vertex_type v, fg.vertices()) {
//     fout << fg.num_neighbors(v) << ", " 
//          << engine.update_count(v)  << std::endl;
//   } // end of foreach vertex update count
//   fout.close();
//   #endif
  
//   //save_maps(fg, mapassg, output_filename);
//   save_beliefs(fg,engine,output_filename);




#include <sill/macros_undef.hpp>
//End of file





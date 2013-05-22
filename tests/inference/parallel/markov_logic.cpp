

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <map>
#include <limits>


#include <sill/model/factor_graph_model.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/factor/log_table_factor.hpp>
#include <sill/inference/parallel/residual_splash_engine.hpp>
#include <sill/inference/parallel/round_robin_engine.hpp>
#include <sill/inference/parallel/basic_state_manager.hpp>
// #include <sill/inference/parallel/average_state_manager.hpp>
#include <boost/program_options.hpp>
#include <sill/parallel/pthread_tools.hpp>
#include <sill/parsers/alchemy.hpp>

#include <sill/macros_def.hpp>


// Rename the program options namespace
namespace po = boost::program_options;

using namespace std;
using namespace sill;

#define ROUND_ROBIN

typedef factor_graph_model<log_table_factor> factor_graph_model_type;
typedef factor_graph_model_type::factor_type factor_type;
typedef factor_graph_model_type::vertex_type vertex_type;
typedef factor_graph_model_type::variable_type variable_type;

typedef basic_state_manager<factor_type> state_manager_type;

#ifdef ROUND_ROBIN
typedef round_robin_engine<factor_type,state_manager_type>
                                                          round_robin_type;
#else
typedef residual_splash_engine<factor_type,state_manager_type>
                                                          residual_splash_type;
#endif



int main(int argc, char* argv[]) {
  string input_filename;
  string output_filename;
  size_t ncpus;
  double bound;
  double damping;
  size_t splashv;

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
    ("ncpus", po::value<size_t>(&ncpus)->default_value(1), 
     "number of cpus")
    ("splashv", po::value<size_t>(&splashv)->default_value(100), 
     "volume of splash")
    ("damping", po::value<double>(&damping)->default_value(0.7), 
     "volume of splash");
  po::positional_options_description pos_opts;
  pos_opts.add("infn",1).add("outfn",1);
  po::variables_map vm;
  store(po::command_line_parser(argc, argv)
        .options(desc).positional(pos_opts).run(), vm);
  notify(vm);
  if(vm.count("help") || !vm.count("infn") || !vm.count("outfn")) {
    cout << "Usage: " << argv[0] << " [options] infn outfn" << endl;
    cout << desc;
    return EXIT_FAILURE;
  }
  
  cout << "==========================================================" << endl
       << "Program Settings: " << endl
       << "Input filename:  " << input_filename << endl
       << "Output filename: " << output_filename << endl
       << "ncpus:           " << ncpus << endl
       << "bound:           " << bound << endl
       << "splash volume:   " << splashv << endl
       << "damping:         " << damping << endl
       << "==========================================================" << endl;

  universe universe;
  factor_graph_model_type fg;

  // Do the actual parsing.
  parse_alchemy(universe, fg, input_filename);
  fg.print_degree_distribution();
  fg.simplify();
  fg.normalize();
  /*
  ofstream fout("adjacency.txt");  
  foreach(const factor_graph_model_type::vertex_type &v, fg.vertices()) {
    int src = fg.vertex2id(v);
    foreach(const factor_graph_model_type::vertex_type &u, fg.neighbors(v)) {
      int dest = fg.vertex2id(u);
      if (src < dest) {
        fout << src << ", " << dest << "\n";
      }
    }
  }
  fout.close();
  */
  cout << "Finished parsing: " << fg.arguments().size() 
       << " variables and " << fg.size() << " factors."
       << endl;

  timer ti;
  ti.start();
  cout << "Performing Inference:" << endl;

  state_manager_type state_manager(&fg, bound, 100);
#ifdef ROUND_ROBIN
  round_robin_type engine(state_manager, 
                              ncpus,
                              bound,
                              damping,false);

#else
  residual_splash_type engine(state_manager, 
                              ncpus,
                              100, 
                              bound,
                              damping,false);
#endif

  finite_assignment mapassg;    
  engine.get_map_assignment(mapassg);
  std::cout.precision(10);
  std::cout << "Log Likelihood: " << fg.log_likelihood(mapassg) << std::endl;


  cout << "Finished!" << endl;
  cout << "Took " << ti.current_time() << "s" << endl;
  return EXIT_SUCCESS;
} // End of main

#include <sill/macros_undef.hpp>
//End of file





// Denoise synthetic target image

// #define SHOW_PROGRESS
// #define DEBUG_OUTPUT

#define TIME_LIMITED
#define TIME_LIMITED_TIME (60*20)



#include <string>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <limits>
#include <cassert>

#include <boost/gil/gil_all.hpp>
#include <boost/gil/extension/io/jpeg_io.hpp>

#include <boost/program_options.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/model/factor_graph_model.hpp>

#include <sill/factor/random/random.hpp>
#include <sill/parallel/pthread_tools.hpp>
#include <sill/parallel/timer.hpp>

#include <sill/inference/parallel/basic_state_manager.hpp>
#include <sill/inference/parallel/basic_update_rule.hpp>
#include <sill/inference/parallel/residual_splash_engine.hpp>

// Denoising image tools
#include <sill/synthetic_data/denoise_image.hpp>


// This should come last
#include <sill/macros_def.hpp>

using namespace std;
using namespace boost::gil;
using namespace sill;

// Declare typedefs 
#ifdef LOGSPACE
typedef table_factor< dense_table<logarithmic<double> > > factor_type;
#else
typedef table_factor factor_type;
#endif

typedef factor_graph_model<factor_type> factor_graph_model_type;
typedef factor_graph_model_type::variable_type variable_type;

typedef basic_state_manager<factor_type> state_manager_type; 

typedef residual_splash_engine<factor_type,state_manager_type>
residual_splash_type;


// Program main
int main(int argc, char** argv) {
  // Parameters to program
  size_t arity;
  size_t rows;
  size_t cols;
  size_t ncpus;
  double bound;
  size_t splash_size;
  double damping;
  double attractive_pot;
  size_t queue_count;
  string truth_image_fname;
  string noisy_image_fname;
  string pred_image_fname;

  // Process command line input
  namespace po = boost::program_options;
  po::options_description 
    desc("Denoise a randomly generated image using ResidualSplash.");
  desc.add_options()
    ("help", "produce help message")
    ("arity", po::value<size_t>(&arity)->default_value(5), 
     "Cardinality of each pixel")
    ("rows", po::value<size_t>(&rows)->default_value(100), 
     "Number of rows in the image")
    ("cols", po::value<size_t>(&cols)->default_value(100), 
     "Number of columns in the image")
    ("ncpus", po::value<size_t>(&ncpus)->default_value(2), 
     "Number of threads to use")
    ("bound", po::value<double>(&bound)->default_value(1.0E-5), 
     "Convergence Bound")
    ("splash", po::value<size_t>(&splash_size)->default_value(50), 
     "Size of each splash")
    ("damping", po::value<double>(&damping)->default_value(0.4),
     "Ammount of damping to use")
    ("pot", po::value<double>(&attractive_pot)->default_value(3.0),
     "The attractive potential")
    ("queue_count", po::value<size_t>(&queue_count)->default_value(100),
     "The number of parallel priority queues for scheduling")
    ("truth_image", po::value<string>(&truth_image_fname), 
     "Filename for true image")
    ("noisy_image", po::value<string>(&noisy_image_fname), 
     "Filename for the noisy iamge")
    ("pred_image", po::value<string>(&pred_image_fname), 
     "Filename for the predicted image");
  
  // Specify the order of ops
  po::positional_options_description pos_opts;
   pos_opts.add("arity",1);
   pos_opts.add("rows",1);
   pos_opts.add("cols",1);
   pos_opts.add("bound",1);
   pos_opts.add("ncpus",1);
   pos_opts.add("splash",1);
   pos_opts.add("damping",1);
  
  po::variables_map vm;
  store(po::command_line_parser(argc, argv)
        .options(desc).positional(pos_opts).run(), vm);
  notify(vm);

  if(vm.count("help") > 0) {
    cout << "Usage: " << argv[0] << "[options]" << endl;
    cout << desc;
    return EXIT_FAILURE;
  }


  // Initialize environment
  universe u;
  srand(time(NULL));   // Initialize the random number generator 
  

/*
  typedef splash_engine<progress_monitor, update_rule,
    LOCKING, LOGSPACE_MESSAGES> managed_cascade_engine_type;
*/
  

  cout << "Initializing distributions: ";
  std::vector<double> mu(arity);
  std::vector<double> var(arity);
  create_distributions(mu,var);
  cout << "Finished." << endl;
  
  cout << "Creating images: ";
  gray32f_image_t noisy_image(rows, cols);
  gray32f_image_t truth_image(rows, cols);
  gray32f_image_t pred_image(rows, cols);
  create_images(mu, var, view(noisy_image), view(truth_image));
  cout << "Finished!" << endl;

  cout << "Creating Graphical Model: ";
  factor_graph_model_type grid_graph; 
  std::vector<variable_type*> variables;
  create_network(u,grid_graph, view(noisy_image), 
                mu, var, attractive_pot, variables);
  cout << "(Simplifying) ";
  grid_graph.simplify();
  cout << "Finished!" << endl;

  if(vm.count("truth_image") > 0 || 
     vm.count("noisy_image") > 0) {
    cout << "Saving input images: ";
    if(vm.count("truth_image") > 0) {
      save_image(truth_image_fname, view(truth_image));
      cout << "(Truth Image) ";
    }
    if(vm.count("noisy_image") > 0) {
      save_image(noisy_image_fname, view(noisy_image));
      cout << "(Noisy Image) ";
    }
    cout << "Finished!" << endl;
  }


  // Start timer
  double running_time = 0.0;
  double error = 0.0;
  timer ti;  ti.start();   
 
  // Do the actual inference
  cout << "Allocating State Manager: " << endl;
  state_manager_type state_manager(&grid_graph, 
                                   bound, 
                                   queue_count);
  cout << "Finished!" << endl;

  cout << "Running Inference:" << endl;
  residual_splash_type engine(state_manager,
                              ncpus,
                              splash_size, 
                              bound,
                              damping,true);
  running_time = ti.current_time();
  cout << "Finished!" << endl;

  if(vm.count("pred_image") > 0) {
    cout << "Saving predicted image: ";
    gray32f_image_t pred_image(rows, cols);
    std::vector<variable_type*> vars;
    foreach(variable_type* v, grid_graph.arguments()) vars.push_back(v);
    create_belief_image(engine, vars, view(pred_image));
    save_image(pred_image_fname, view(pred_image));
    cout << "Finished:" << endl;
  }
 
  // Printing results for further processing
  cout << "Output!: "
       << arity << "\t"
       << rows << "\t" 
       << cols << "\t" 
       << bound << "\t"
       << ncpus << "\t"
       << splash_size << "\t"
       << running_time << "\t" 
       << error << endl;


  return (EXIT_SUCCESS);
} 


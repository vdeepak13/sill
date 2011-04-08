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

#include <boost/random/lagged_fibonacci.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/gil/gil_all.hpp>
#include <boost/gil/extension/io/jpeg_io.hpp>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

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
#include <sill/inference/parallel/mpi_residual_splash_engine.hpp>
#include <sill/inference/parallel/mpi_state_manager.hpp>
#include <sill/inference/parallel/mpi_state_manager_protocol.hpp>

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
typedef tablef factor_type;
#endif

typedef factor_graph_model<factor_type> factor_graph_model_type;
typedef factor_graph_model_type::variable_type variable_type;

typedef mpi_state_manager<factor_type> state_manager_type; 

typedef mpi_residual_splash_engine<factor_type,state_manager_type>
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
  namespace popt = boost::program_options;
  popt::options_description 
    desc("Denoise a randomly generated image using ResidualSplash.");
  desc.add_options()
    ("help", "produce help message")
    ("arity", popt::value<size_t>(&arity)->default_value(5), 
     "Cardinality of each pixel")
    ("rows", popt::value<size_t>(&rows)->default_value(100), 
     "Number of rows in the image")
    ("cols", popt::value<size_t>(&cols)->default_value(100), 
     "Number of columns in the image")
    ("ncpus", popt::value<size_t>(&ncpus)->default_value(2), 
     "Number of threads to use")
    ("bound", popt::value<double>(&bound)->default_value(1.0E-5), 
     "Convergence Bound")
    ("splash", popt::value<size_t>(&splash_size)->default_value(50), 
     "Size of each splash")
    ("damping", popt::value<double>(&damping)->default_value(0.4),
     "Ammount of damping to use")
    ("pot", popt::value<double>(&attractive_pot)->default_value(3.0),
     "The attractive potential")
    ("queue_count", popt::value<size_t>(&queue_count)->default_value(100),
     "The number of parallel priority queues for scheduling")
    ("truth_image", popt::value<string>(&truth_image_fname), 
     "Filename for true image")
    ("noisy_image", popt::value<string>(&noisy_image_fname), 
     "Filename for the noisy iamge")
    ("pred_image", popt::value<string>(&pred_image_fname), 
     "Filename for the predicted image");
  
  // Specify the order of ops
  popt::positional_options_description pos_opts;
   pos_opts.add("arity",1);
   pos_opts.add("rows",1);
   pos_opts.add("cols",1);
   pos_opts.add("bound",1);
   pos_opts.add("ncpus",1);
   pos_opts.add("splash",1);
   pos_opts.add("damping",1);
  
  popt::variables_map vm;
  store(popt::command_line_parser(argc, argv)
        .options(desc).positional(pos_opts).run(), vm);
  notify(vm);

  if(vm.count("help") > 0) {
    cout << "Usage: " << argv[0] << "[options]" << endl;
    cout << desc;
    return EXIT_FAILURE;
  }

  mpi_post_office po;
  factor_graph_model_type grid_graph; 
  mpi_state_manager<tablef> *state; 
    universe u;
  if (po.id() == 0) {
    // Initialize environment
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
    std::vector<variable_type*> variables;
    std::vector<factor_type*> factors;
    create_network(u,grid_graph, view(noisy_image), 
                  mu, var, attractive_pot, variables, factors);
    cout << "(Simplifying) ";
    grid_graph.simplify();
    cout << "Finished!" << endl;
    // exit(0);
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
    
    state = new mpi_state_manager<tablef>(po,     // post office
                                      &grid_graph,    // factor graph
                                      1.0E-5, // epsilon
                                      5000);   // max vertices per node
  }
  else {
    state = new mpi_state_manager<tablef>(po);
  }

  cout << "Running Inference:" << endl;
  
    po.start();
  std::cout << "MPI Started!\n";
  state->start();
  
    // Start timer
  double running_time = 0.0;
  double error = 0.0;
  timer ti;  ti.start();   

  residual_splash_type engine(*state,
                            1,
                            splash_size, 
                            bound,
                            damping,
                            false);
  std::cout << "FIN!\n";
  MPI::COMM_WORLD.Barrier();
  if (po.id() == 0) {
    state->collect_beliefs();
    std::cout << "Received Beliefs\n";
  }
  MPI::COMM_WORLD.Barrier();
  
  running_time = ti.current_time();
  cout << "Finished!" << endl;
  if (po.id() == 0) {
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
    po.stopAll();
  }
  po.wait();
  return (EXIT_SUCCESS);
} 












// Denoise synthetic target image using mpi library



#include <string>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <limits>
#include <cassert>

#include <mpi.h>

#include <boost/gil/gil_all.hpp>
#include <boost/gil/extension/io/jpeg_io.hpp>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <boost/program_options.hpp>

#include <prl/base/universe.hpp>
#include <prl/factor/table_factor.hpp>
#include <prl/model/factor_graph_model.hpp>

#include <prl/factor/random.hpp>
#include <prl/parallel/pthread_tools.hpp>
#include <prl/parallel/timer.hpp>


#include <prl/mpi/sync/mpi_sync_engine.hpp>


// Denoising image tools
#include <prl/synthetic_data/denoise_image.hpp>


// This should come last
#include <prl/macros_def.hpp>

using namespace std;
using namespace boost::gil;
using namespace prl;

// Declare typedefs 
typedef tablef factor_type;

typedef factor_graph_model<factor_type> factor_graph_model_type;
typedef factor_graph_model_type::variable_type variable_type;

template<typename T>
void mpi_bcast(T& item) {
  MPI::COMM_WORLD.Bcast(reinterpret_cast<char*>(&item),
                        sizeof(T), 
                        MPI::CHAR, 
                        0);
}


int root_node(int argc, char** argv) {

  // Parameters to program
  size_t arity;
  size_t rows;
  size_t cols;
  double bound;
  size_t splash_size;
  size_t fringe_size;
  double damping;
  double attractive_pot;
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
    ("bound", po::value<double>(&bound)->default_value(1.0E-5), 
     "Convergence Bound")
    ("splash", po::value<size_t>(&splash_size)->default_value(50), 
     "Size of each splash")
    ("fringe", po::value<size_t>(&fringe_size)->default_value(5), 
     "height of fringe")
    ("damping", po::value<double>(&damping)->default_value(0.4),
     "Ammount of damping to use")
    ("pot", po::value<double>(&attractive_pot)->default_value(3.0),
     "The attractive potential")
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
   pos_opts.add("splash",1);
   pos_opts.add("fringe",1);
   pos_opts.add("damping",1);
  
  po::variables_map vm;
  store(po::command_line_parser(argc, argv)
        .options(desc).positional(pos_opts).run(), vm);
  notify(vm);

  if(vm.count("help") > 0) {
    cout << "Usage: " << argv[0] << "[options]" << endl;
    cout << desc;
    bool bad_input = true;
    mpi_bcast(bad_input);
    return EXIT_FAILURE;
  }

  bool bad_input = false;
  mpi_bcast(bad_input);
  mpi_bcast(bound);
  mpi_bcast(splash_size);
  mpi_bcast(fringe_size);
  mpi_bcast(damping);

 
  // Initialize environment
  universe u;
  srand(time(NULL));   // Initialize the random number generator 
    
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
  factor_graph_model_type factor_graph; 
  std::vector<variable_type*> variables;
  std::vector<factor_type*> factors;
  create_network(u, factor_graph, view(noisy_image), 
                 mu, var, attractive_pot, variables, factors);
  cout << "(Simplifying) ";
  factor_graph.simplify();
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
  
  std::cout << "Sending Graph" << std::endl;
  mpi_sync_engine engine(&factor_graph,
                         splash_size,
                         fringe_size,
                         bound,
                         damping);
  std::cout << "Finished Sending Graph" << std::endl;
  
  // Start timer
  double running_time = 0.0;
  double error = 0.0;
  timer ti;  ti.start();   
  cout << "Running Inference on root:" << endl;
  engine.start();
  running_time = ti.current_time();
  std::cout << "Finished!" << std::endl;

  if(vm.count("pred_image") > 0) {
    cout << "Saving predicted image: ";
    gray32f_image_t pred_image(rows, cols);
    std::vector<variable_type*> vars;
    foreach(variable_type* v, factor_graph.arguments()) vars.push_back(v);
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
       << splash_size << "\t"
       << running_time << "\t" 
       << error << endl;
 
  return EXIT_SUCCESS;
}


int slave_node() {
  double bound = 0.0;
  size_t splash_size = 0;
  size_t fringe_size = 0;
  double damping = 0.0;
  bool bad_input = false;

  mpi_bcast(bad_input);
  if(bad_input) return EXIT_FAILURE;

  //  std::cout << "Receiving Parameters:" << std::endl;
  mpi_bcast(bound);
  mpi_bcast(splash_size);
  mpi_bcast(fringe_size);
  mpi_bcast(damping);
  
  cout << "Receiving Graph:" << endl;
  mpi_sync_engine engine(NULL,
                         splash_size, 
                         fringe_size,
                         bound,
                         damping);
  cout << "Running Inference:" << endl;
  engine.start();
  cout << "Finished!" << std::endl;  
  return EXIT_SUCCESS;
}


// Program main
int main(int argc, char** argv) {
  int exit_status = EXIT_SUCCESS;

  // Startup MPI
  MPI::Init();

  int id = MPI::COMM_WORLD.Get_rank();

  if(id == 0) {
    int ret = root_node(argc, argv);
    if(ret == EXIT_FAILURE) exit_status = EXIT_FAILURE;
  } else {
    int ret = slave_node();
    if(ret == EXIT_FAILURE) exit_status = EXIT_FAILURE;
  }

  MPI::Finalize();
  //  std::cout << "Beginning Wait" << std::endl;
  return exit_status;
} 


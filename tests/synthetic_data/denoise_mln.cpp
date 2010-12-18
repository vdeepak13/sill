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

#include <sill/parsers/alchemy.hpp>

// Denoising image tools
#include <sill/synthetic_data/denoise_image.hpp>


// This should come last
#include <sill/macros_def.hpp>

using namespace std;
using namespace boost::gil;
using namespace sill;

// Declare typedefs 
// typedef table_factor< dense_table<logarithmic<double> > > factor_type;
typedef table_factor factor_type;
typedef factor_graph_model<factor_type> factor_graph_type;
typedef factor_graph_type::vertex_type vertex_type;
typedef factor_graph_type::variable_type variable_type;


// Program main
int main(int argc, char** argv) {
  // Parameters to program
  size_t arity;
  size_t rows;
  size_t cols;
  double attractive_pot;

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
    ("pot", po::value<double>(&attractive_pot)->default_value(3.0),
     "The attractive potential");

  // Specify the order of ops
  po::positional_options_description pos_opts;
  pos_opts.add("arity",1);
  pos_opts.add("rows",1);
  pos_opts.add("cols",1);
  
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
  srand(100);   // Initialize the random number generator   

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
  factor_graph_type fg;
  std::vector<variable_type*> variables;
  create_network(u, fg, view(noisy_image), 
                 mu, var, attractive_pot, variables);
  
  cout << "(Simplifying) ";
  fg.simplify_stable();
  fg.normalize();
  cout << "Finished!" << endl;

  cout << "Saving raw images" << endl;
//  save_image("true.jpg", view(truth_image));
//  save_image("noisy.jpg", view(noisy_image));
  
  cout << "Saving graph info" << endl;
  std::ofstream fout;  
  fout.open("adjacency.txt");  
  fg.print_adjacency(fout);
  fout.close();

  fout.open("vertinfo.txt");
  fg.print_vertex_info(fout);
  fout.close();

  cout << "Saving factor graph to file" << std::endl;
  fout.open("factorgraph.out");
  print_alchemy(fg, fout);
  fout.close();


  return (EXIT_SUCCESS);
} 



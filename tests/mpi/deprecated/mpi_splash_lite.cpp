
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <map>

#include <mpi.h>

#include <prl/factors/table_factor.hpp>
#include <prl/model/factor_graph_model.hpp>
#include <prl/datastructures/mutable_queue.hpp>

using namespace std;
using namespace prl;

// typedefs
 
typedef tablef factor_type;
typedef factor_type::variable_type      variable_type;
typedef factor_graph_model<factor_type> factor_graph_type;
typedef factor_graph_type::vertex_type  vertex_type;

typedef int machine_id;

typedef map<vertex_type, machine_id>        ownership_type;
typedef mutable_queue<vertex_type, double>  queue_type;
typedef pair<vertex_type, vertex_type>      directed_edge;
typedef map<directed_edge, factor_type >    message_map_type;
typedef map<vertex_type, factor_type>       belief_map_type;

typedef map<variable_type*, uint32_t> var2id_map_type;
typedef map<uint32_t, variable_type*> id2var_map_type;


// Populate the target image
void populate_images(vector<double> mu,
                     vector<double> var,
                     const gray32f_view_t& noisy_image, 
                     const gray32f_view_t& truth_image) {
   
  // Define parameters
  size_t rows = noisy_image.height();
  size_t cols = noisy_image.width();
  size_t arity = mu.size();

  // Create sources of randomness
  boost::lagged_fibonacci607 randomness;
  std::vector< boost::normal_distribution<double> > dist(arity);
  for(size_t i = 0; i < arity; ++i) 
    dist[i] = boost::normal_distribution<double>(mu[i], sqrt(var[i]));

  // Compute image center
  const double center_r = rows/2.0;
  const double center_c = cols/2.0;
  const double max_radius = std::min(rows, cols) / 2.0;

  // Fill out the image
  for(size_t r = 0; r < rows; ++r) {
    for(size_t c = 0; c < cols; ++c) {
      double distance = sqrt((r-center_r)*(r-center_r) + 
                             (c-center_c)*(c-center_c));
      // Compute ring
      size_t ring = static_cast<size_t>(floor(std::min(1.0, distance/max_radius)
                                              * (arity - 1) ) );
      assert(ring >= 0 && ring < mu.size());
      if(r < rows / 2) {
        noisy_image(c,r) = dist[ring](randomness);
        truth_image(c,r) = ring;
      } else {
        noisy_image(c,r) = dist[0](randomness);
        truth_image(c,r) = 0;
      }
    }
  }
} // end of populate images



/**
 * variables and factors should be empty and will be filled 
 * in and returned by this fucntion
 */
factor_graph_type* 
create_model(universe& u,
             size_t arity,
             size_t rows,
             size_t cols,
             double bw) {
  // Create distributions parameters
  vector<double> mu(arity), var(arity);
  for(size_t i = 0; i < mu.size(); ++i) {
    mu[i]  = i * 7;
    var[i] = 10*10;
  }

  // Create an images
  gray32f_image_t truth_image(rows, cols);
  gray32f_view_t  truth_view(truth_image);
  gray32f_image_t noisy_image(rows, cols);
  gray32f_view_t  noisy_view(noisy_image);
  populate_image(mu, var, noisy_view, truth_view);


  // allocate a factor graph
  factor_graph_type* factor_graph = new factor_graph_type();
  
  // Construct the set of variables
  vector<variable_type*> variables;
  variables.resize(rows*cols);
  for (size_t i = 0; i < rows * cols; ++i) {
    char c[16];
    sprintf(c,"%lu",i);
    variables[i] = u.new_finite_variable(c, cardinality);
  }

  
  // Compute the node factors
  for(size_t r = 0; r < rows; ++r) {
    for(size_t c = 0; c < cols; ++c) {
      // Compute the variable id from the row x col index
      size_t u = r * cols + c;
      finite_domain args;
      args.insert(variables[u]);
      factor_type factor_type(args, 1.0);
      for(size_t asg = 0; asg < arity; ++asg){
        factor(asg) =
          exp (-(noisy_view(c,r)[0] - mu[asg])*(noisy_view(c,r)[0] - mu[asg]) / 
               (2.0 * var[asg]));
      }
      factor_graph.add_factor(factor);
    }
  } // end of loop over node factors
  
  // Add the edge factors 
  for(size_t r = 0; r < rows; ++r) {
    for(size_t c = 0; c < cols; ++c) {
      size_t u1 = r * cols + c;
      // Add the horizontal factor if one exists
      if(c < cols-1) {
        finite_domain args = make_domain(variables[u1], variables[u1+1]);
        factor_type factor(args, exp(-bw));
        for(size_t asg = 0; asg < cardinality; ++asg){  
          factor(asg,asg) = 1.0;
        }
        factor_graph.add_factor(factor);
      } // end of if horizontal
      // Add the vertical factor if one exists
      if(r < rows - 1) {
        finite_domain args = make_domain(variables[u1], variables[u1+cols]);
        factor_type factor(args, exp(-bw));
        for(size_t asg = 0; asg < cardinality; ++asg){  
          factor(asg,asg) = 1.0;
        }
        factor_graph.add_factor(factor);
      } // end of if vertical
    }
  } // end of add edge factors
  return factor_graph;
}




// If this is process zero then build the factor graph, broadcast the
// factor graph, and return a pointer
factor_graph_type* get_factor_graph(int id,
                                    universe& u, 
                                    size_t rows,
                                    size_t cols,
                                    double bw) {
  factor_graph_type* factor_graph;
  if (id == 0) {
    factor_graph = create_model(u, rows, cols, bw);
    // send var2id and id2var maps
    // vertex2owner and owner2vertex
    std::stringstream strm;
    strm.write(reinterpret_cast<const char*>(&typeofmessage), sizeof(int32_t));
    boost::archive::binary_oarchive arc(strm);
    arc << *(state_.model_);
    // MPI::Send(strm.str().c_str()....)
  }
  else {
    // MPI::RECEIVE body, bodylen
    boost::iostreams::stream<boost::iostreams::array_source> strm(body, bodylen);
    boost::archive::binary_iarchive arc(strm);
    ret = new typename factor_graph_model_type;
    arc >> *ret;
  }
} // end of get factor graph

// If this is process zero then build the ownership set, broadcast the
// ownership and return a pointer
ownership_type* get_ownership(const factor_graph_type* fg, int id) {


} // end of get ownwership


// Runs the splash algorithm
void splash_loop(int id,
                 factor_graph_type* fg, 
                 ownership_type* ownership,
                 size_t splashsize,
                 double alpha) {
  // Initialize the prirority queue by adding all vertices 
  // that this processor owns
  queue_type queue;
  foreach(queue_type::elem_type elem, ownership) {
    if(elem.second == id) 
      queue.add(id, numeric_limits<double>::max_value());
  }
  
  // Initialize a message map
  message_map_type messages;
  list<factor_type> out_messages;

  bool active = true;
  while(active) {
    // get the top vertex
    vertex v = queue.dequeue();
    // do a splash updating messages and out_messages
    splash(v, fg, messages, out_messages);
    // Send any outbound messages




  } // End of while loop
} // end of splash_loop
 

int main(int argc, char* argv[]) {
  // Initialize MPI
  MPI::COMM_WORLD.Init();
  int id = MPI::COMM_WORLD.Get_rank();  
  size_t numprocs = MPI::COMM_WORLD.Get_size();
  
  // Using bcast get the factor graph and ownership information for
  // this processor id
  factor_graph_type* factor_graph = get_factor_graph(id);
  ownership_type* ownership = get_ownership(factor_graph, id);

  // Create a single threaded mpi_splash_engine
  size_t splash_size = 100;
  double damping = 0.4;

  // This will block until inference is complet
  if(id == 0) {
    termination_manager();
  } else {
    splash_loop(id, factor_graph, ownership, splash_size, damping);
  }
  // Finalize the mpi environment
  MPI::Finalize();
  // Create an mpi_inf
  return EXIT_SUCCESS;
}

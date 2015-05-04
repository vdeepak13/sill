#ifndef BINARY_RESIDUAL_SPLASH_BP_HPP
#define BINARY_RESIDUAL_SPLASH_BP_HPP


// STL includes
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <sstream>
#include <limits>

// SILL Includes
#include <sill/model/factor_graph_model.hpp>
#include <sill/factor/util/norms.hpp>
#include <sill/math/gdl_enum.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/datastructure/mutable_queue.hpp>



// This include should always be last
#include <sill/macros_def.hpp>
namespace sill {

  template<typename F>
  class binary_residual_splash_bp {
    ///////////////////////////////////////////////////////////////////////
    // typedefs
  public:
    typedef F factor_type;
    typedef factor_type message_type;
    typedef factor_type belief_type;
    typedef typename factor_type::domain_type domain_type;

    typedef factor_graph_model<factor_type>     factor_graph_type;
    typedef typename factor_graph_type::variable_type    variable_type;
    typedef typename factor_graph_type::vertex_type      vertex_type;

    typedef std::set<vertex_type> vertex_set_type;
    typedef factor_norm_1<message_type> norm_type;

    typedef std::vector<vertex_type> ordering_type;

    
    
    //! IMPORTANT map is in [dest][src] form
    typedef std::map<vertex_type, 
                     std::map<vertex_type, message_type> > 
    message_map_type;

   
    ///////////////////////////////////////////////////////////////////////
    // Data members
  private:    
    //! pointer to the factor graph 
    factor_graph_type* factor_graph_;
    
    //! the schedule of vertices
    std::set<vertex_type>  zeros_;
    std::list<vertex_type>  ones_;
    std::map<vertex_type, double> residuals_;
    //! messages
    message_map_type messages_;
   
    //! the size of a splash
    size_t splash_size_;

    //! convergence bound
    double bound_;

    //! level of damping 1.0 is fully damping and 0.0 is no damping
    double damping_;

    //! the commutative semiring for updates (typically sum_product)
    commutative_semiring csr_;

  public:

    /**
     * Create a residual splash engine
     */
    binary_residual_splash_bp(factor_graph_type* factor_graph,
                       size_t splash_size,
                       double bound,
                       double damping) :
      factor_graph_(factor_graph), 
      splash_size_(splash_size), 
      bound_(bound), 
      damping_(damping),
      csr_(sum_product) {
      // Ensure that the factor graph is not null
      assert(factor_graph_ != NULL);
    } // end residual_splash_bp


    /**
     * Executes the actual engine. 
     */
    void run() {
      // initialize the messages
      initialize_state();
      // Ran splash to convergence
      splash_to_convergence();
    } // End of run
    

    /**
     * This function preallocates messages and initializes the priority queue
     */
    void initialize_state() {
      // Clear the messages
      messages_.clear();
      // Allocate all messages
      foreach(const vertex_type& u, factor_graph_->vertices()) {
        foreach(const vertex_type& v, factor_graph_->neighbors(u)) { 
          message_type& msg = messages_[u][v];
          domain_type domain = make_domain(u.is_variable() ? 
                                           &(u.variable()) :
                                           &(v.variable()));
          msg = message_type(domain, 1.0).normalize();
        }
      }
      // Initialize the schedule
      zeros_.clear();
      std::vector<vertex_type> tempvec;
      ones_.clear();
      // double initial_residual = bound_ + (bound_*bound_);
      double initial_residual = std::numeric_limits<double>::max();
      foreach(const vertex_type& u, factor_graph_->vertices()) {
        tempvec.push_back(u);
        residuals_[u] = initial_residual;
      }
      random_shuffle(tempvec.begin(),tempvec.end());
      foreach(const vertex_type& u, tempvec) {
        ones_.push_back(u);
      }
    } // end of initialize


    /**
     * Splash to convergence
     */
    void splash_to_convergence() {
      timer finishtimer;
      finishtimer.start();
      double lasthittime = finishtimer.current_time();
      
      size_t update_count = 0;
      while(ones_.size() > 0) {
        vertex_type v = ones_.front();
        ones_.pop_front();
        zeros_.insert(v);
        splash(v);
        update_count++;
        if(finishtimer.current_time() - lasthittime > 5) {
          lasthittime = finishtimer.current_time();
          std::cout << ones_.size() << " " << update_count << std::endl;
        }
      }
      std::cout << update_count << " updates\n";
    }


    /**
     * Given a vertex this computes a single splash around that vertex
     */
    void splash(const vertex_type& root) {
      ordering_type splash_order;
      // Grow a splash ordering
      generate_splash(root, splash_order);          
      // Push belief from the leaves to the root
      revforeach(const vertex_type& v, splash_order) {
        send_messages(v);
      }
      // Push belief from the root to the leaves (skipping the
      // root which was processed in the previous pass)
      foreach(const vertex_type& v, 
              std::make_pair(++splash_order.begin(), splash_order.end())) {
        send_messages(v);
      }
    } // End of splash_once


    /**
     * This function computes the splash ordering (a BFS) search for
     * the root vertex
     */
    void generate_splash(const vertex_type& root, 
                         ordering_type& splash_order) {
      typedef std::set<vertex_type> visited_type;
      typedef std::list<vertex_type> queue_type;
      // Create a set to track the vertices visited in the traversal
      visited_type visited;
      queue_type splash_queue;
      // Set the root to be visited and the first element in the queue
      splash_queue.push_back(root);
      visited.insert(root);
      // Grow a breath first search tree around the root      
      for(size_t i = 0; i < splash_size_ && !splash_queue.empty(); ++i) {
        // Remove the first element
        vertex_type u = splash_queue.front();
        splash_queue.pop_front();
        // Insert the first element into the tree order
        splash_order.push_back(u);
        // If we need more vertices then grow out more
        if(splash_order.size() + splash_queue.size() < splash_size_) {
          // Add all the unvisited neighbors to the queue
          foreach(const vertex_type& v, factor_graph_->neighbors(u)) {
            if((visited.count(v) == 0) && residuals_[v] > bound_) {      
              splash_queue.push_back(v);
              visited.insert(v);
            }
          } // end of for each neighbors
        } // End of if statement
      } // End of foor loop
    } // End of Generate Splash



    /**
     * This function computes the splash ordering (a BFS) search for
     * the root vertex
     */
    void generate_splash_priority(const vertex_type& root, 
                         ordering_type& splash_order) {
      typedef std::set<vertex_type> visited_type;
      typedef mutable_queue<vertex_type,double> queue_type;
      // Create a set to track the vertices visited in the traversal
      visited_type visited;
      queue_type splash_queue;
      // Set the root to be visited and the first element in the queue
      splash_queue.push(root, 100);
      visited.insert(root);
      // Grow a breath first search tree around the root      
      for(size_t i = 0; i < splash_size_ && !splash_queue.empty(); ++i) {
        // Remove the first element
        vertex_type u = splash_queue.top().first;
        splash_queue.pop();
        // Insert the first element into the tree order
        splash_order.push_back(u);
        // If we need more vertices then grow out more
        if(splash_order.size() + splash_queue.size() < splash_size_) {
          // Add all the unvisited neighbors to the queue
          foreach(const vertex_type& v, factor_graph_->neighbors(u)) {
            if((visited.count(v) == 0) && residuals_[v] > bound_) {      
              splash_queue.push(v, residuals_[v]);
              visited.insert(v);
            }
          } // end of for each neighbors
        } // End of if statement
      } // End of foor loop
    } // End of Generate Splash


    /**
     * This writes the new message into the place of the old message
     * and updates the scheduling queue and does any damping necessary
     */
    inline void update_message(const vertex_type& source,
                               const vertex_type& target,
                               const message_type& new_msg) {
      // Create a norm object (this should be lightweight) 
      norm_type norm;
      // Get the original message
      message_type& original_msg = messages_[source][target];
      // Compute the norm
      double new_residual = norm(new_msg, original_msg);
      // Update the residual
      if(residuals_[target] < new_residual ) {
        residuals_[target] = new_residual;
        if (new_residual > bound_) {
          typename set<vertex_type>::iterator i = zeros_.find(target);
          if (i!=zeros_.end()) {
            zeros_.erase(i);
            ones_.push_back(target);
          }
        }
      }
      // Require that there be no zeros
      // assert(new_msg.minimum() > 0.0);
      // Save the new message
      original_msg = new_msg;
    } // end of update_message


    /**
     * Receive all messages into the vertex and compute all new
     * outbound messages.
     */
    inline void send_messages(const vertex_type& source) {
      // For each of the neighboring vertices (vertex_target) of
      // vertex compute the message from vertex to vertex_target
      foreach(const vertex_type& dest, 
              factor_graph_->neighbors(source)) {
        send_message(source, dest);
      }
      // Mark the vertex as having been visited
      residuals_[source] = 0;
    } // end of update messages



    /**
     * Send the message from vertex_source to vertex_target.  Note
     * that if another processor is currently trying to send this
     * message then this routine will simply return;
     */
    inline void send_message(const vertex_type& source,
                             const vertex_type& target) {
      

      message_type new_msg;

      if(source.is_variable()) {
        // create a temporary message to store the result into
        domain_type domain = make_domain(source.is_variable() ? 
                                         &(source.variable()) :
                                         &(target.variable()));
        new_msg = message_type(domain, 1.0).normalize();
        // If the source was a factor we multiply in the factor potential
      } else {
        // Set the message equal to the factor.  This will increase
        // the size of the message and require an allocation.
        new_msg = source.factor();
      } 

      // For each of the neighbors of the vertex
      foreach(const vertex_type& other, 
              factor_graph_->neighbors(source)) {
        // if this is not the dest_v
        if(other != target) {          
          // Combine the in_msg with the destination factor
          new_msg.combine_in( messages_[other][source], csr_.dot_op);
          // Here we normalize after each iteration for numerical
          // stability.  This could be very costly for large factors.
          new_msg.normalize();
        }
      }        
      // If this is a message from a factor to a variable then we
      // must marginalize out all variables except the the target
      // variable.  
      if(source.is_factor()) {
        new_msg = new_msg.collapse(csr_.cross_op, make_domain(&target.variable()));
      }
      // Normalize the message
      new_msg.normalize();
      // Damp messages form factors to variables
      if(target.is_variable()) {
        new_msg = weighted_update(new_msg, 
                                  messages_[source][target], 
                                  damping_);
      }
      // update the message also updating the schedule
      update_message(source, target, new_msg);
    } // end of send_message


    /**
     * Compute the belief for a vertex
     */
    belief_type belief(variable_type variable) {
      // Initialize the belief as uniform
      belief_type blf = belief_type(make_domain(variable), 
                                    1.0).normalize();
      vertex_type vertex(variable);
      // For each of the neighbors of the vertex
      foreach(const vertex_type& other, 
              factor_graph_->neighbors(vertex)) {
        // Combine the in_msg with the destination factor
        blf.combine_in( messages_[other][vertex], csr_.dot_op);
        // Here we normalize after each iteration for numerical
        // stability.  This could be very costly for large factors.
       blf.normalize();
      }        
      // Normalize the message
      blf.normalize();
      // Return the final belief 
      return blf;
    } // end of send_message


  }; // End of class residual_splash_bp


}; // end of namespace
#include <sill/macros_undef.hpp>



#endif

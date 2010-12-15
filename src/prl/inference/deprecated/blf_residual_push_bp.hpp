#ifndef BLF_RESIDUAL_PUSH_BP_HPP
#define BLF_RESIDUAL_PUSH_BP_HPP


// STL includes
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <sstream>
#include <limits>

// PRL Includes
#include <prl/model/factor_graph_model.hpp>
#include <prl/factor/norms.hpp>
#include <prl/math/gdl_enum.hpp>
#include <prl/factor/table_factor.hpp>
#include <prl/datastructure/mutable_queue.hpp>



// This include should always be last
#include <prl/macros_def.hpp>
namespace prl {

  template<typename F>
  class blf_residual_push_bp {
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

    typedef mutable_queue<vertex_type, double> schedule_type;

    
    //! IMPORTANT map is in [dest][src] form
    typedef std::map<vertex_type, 
                     std::map<vertex_type, message_type> > 
    message_map_type;

    typedef std::map<vertex_type, belief_type> belief_map_type;
   
    ///////////////////////////////////////////////////////////////////////
    // Data members
  private:    
    //! pointer to the factor graph 
    factor_graph_type* factor_graph_;
    
    //! the schedule of vertices
    schedule_type schedule_;
    
    //! messages
    message_map_type messages_;

    //! beliefs
    belief_map_type last_beliefs_;
    belief_map_type beliefs_;
   
    //! the size of a splash
    size_t splash_size_;

    //! convergence bound
    double bound_;

    //! level of damping 1.0 is fully damping and 0.0 is no damping
    double damping_;

    //! the commutative semiring for updates (typically sum_product)
    commutative_semiring csr_;

  public:




    blf_residual_push_bp(factor_graph_type* factor_graph,
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
      std::cout << "created blf_residual_push_bp engine" << std::endl;
    } // end blf_residual_push_bp

    void run() {
      // initialize the messages
      initialize_state();
      // Ran splash to convergence
      splash_to_convergence();
    } // End of run
    

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
        // Initialize the belief
        belief_type& last_blf = last_beliefs_[u];
        belief_type& blf = beliefs_[u];
        if(u.is_factor()) { 
          blf = u.factor();
          last_blf = blf;
        } else {
          blf = belief_type(make_domain(&(u.variable())),1.0).normalize();
          last_blf = blf;
        }
      }
      // Initialize the schedule
      schedule_.clear();
      // double initial_residual = bound_ + (bound_*bound_);
      double initial_residual = std::numeric_limits<double>::max();
      foreach(const vertex_type& u, factor_graph_->vertices()) {
        schedule_.push(u, initial_residual);
      }
    } // end of initialize




    void splash_to_convergence() {
      assert(schedule_.size() > 0);
      size_t update_count = 0;
      while(schedule_.top().second > bound_) {
        splash(schedule_.top().first);
        if((update_count++ % 100) == 0) {
          std::cout << schedule_.top().second << std::endl;
        }
      }
    } // end of splash_to_convergence



    void splash(const vertex_type& root) {
      typedef std::set<vertex_type> in_tree_type;
      typedef std::list<vertex_type> queue_type;
      // Create a set to track the vertices visited in the traversal
      in_tree_type in_tree;
      queue_type queue;
      // Set the root to be visited and the first element in the queue
      queue.push_back(root);
      in_tree.insert(root);
      // Grow a breath first search tree around the root      
      for(size_t work_done = 0; 
          work_done < splash_size_ && !queue.empty();) {
        // Remove the first element
        vertex_type u = queue.front();
        queue.pop_front();
        // Send all the messages
        send_messages(u);
        // Account the work done 
        work_done += factor_graph_->num_neighbors(u);
        // If we need more vertices then grow out more
        if(work_done + queue.size() < splash_size_) {
          // Add all the unvisited neighbors to the queue
          foreach(const vertex_type& v, factor_graph_->neighbors(u)) {
            if((in_tree.count(v) == 0) && schedule_.get(v) > bound_) {      
              queue.push_back(v);
              in_tree.insert(v);
            }
          } // end of for each neighbors
        } // End of if statement
      } // End of foor loop

      std::cout << "Tree Size: " << in_tree.size() << std::endl;
    } // End of splash_once




    inline void update_message(const vertex_type& source,
                               const vertex_type& target,
                               const message_type& new_msg) {
      // Create a norm object (this should be lightweight) 
      norm_type norm;
      // Get the original message
      message_type& original_msg = messages_[source][target];
      belief_type& blf = beliefs_[target];
      blf.combine_in(original_msg, divides_op);
      blf.combine_in(new_msg, csr_.dot_op);
      blf.normalize();
      original_msg = new_msg;
      double new_residual = norm(blf, last_beliefs_[target]);
      // Update the residual
      if(schedule_.get(target) < new_residual )
        schedule_.update(target, new_residual);
    } // end of update_message






    inline void send_messages(const vertex_type& source) {
      // For each of the neighboring vertices (vertex_target) of
      // vertex compute the message from vertex to vertex_target
      foreach(const vertex_type& dest, 
              factor_graph_->neighbors(source)) {
        send_message(source, dest);
      }
      // Mark the vertex as having been visited
      schedule_.update(source, 0.0);
      // update the belief
      last_beliefs_[source] = beliefs_[source];
    } // end of update messages



    inline void send_message(const vertex_type& source,
                             const vertex_type& target) {
      // here we can assume that the current belief is correct
      belief_type& blf = beliefs_[source];

      // Construct the cavity
      belief_type cavity = combine(blf, 
                                   messages_[target][source], 
                                   divides_op);
      // Marginalize out any other variables
      domain_type domain = make_domain(source.is_variable()?
                                       &(source.variable()) :
                                       &(target.variable()));
      message_type new_msg = cavity.collapse(domain, csr_.cross_op);
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
    belief_type belief(variable_type* variable) {
      return beliefs_[vertex_type(variable)];
    } // end of send_message

  }; // End of class blf_residual_push_bp


}; // end of namespace
#include <prl/macros_undef.hpp>



#endif

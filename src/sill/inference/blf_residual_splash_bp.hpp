#ifndef BLF_RESIDUAL_SPLASH_BP_HPP
#define BLF_RESIDUAL_SPLASH_BP_HPP

#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>

#include <sill/model/factor_graph_model.hpp>
#include <sill/factor/norms.hpp>
#include <sill/inference/commutative_semiring.hpp>
#include <sill/datastructure/mutable_queue.hpp>

// This include should always be last
#include <sill/macros_def.hpp>

namespace sill {

  template<typename F>
  class blf_residual_splash_bp {
    ///////////////////////////////////////////////////////////////////////
    // typedefs
  public:
    typedef F factor_type;
    typedef F message_type;
    typedef F belief_type;
    typedef typename F::domain_type domain_type;

    typedef factor_graph_model<factor_type>     factor_graph_type;
    typedef typename factor_graph_type::variable_type    variable_type;
    typedef typename factor_graph_type::vertex_type      vertex_type;

    typedef std::set<vertex_type> vertex_set_type;



    typedef std::vector<vertex_type> ordering_type;

    typedef mutable_queue<vertex_type, double> schedule_type;

    
    
    typedef std::map<vertex_type, 
                     std::map<vertex_type, message_type> > 
    message_map_type;

    typedef std::map<vertex_type, belief_type> belief_map_type;

    typedef std::map<vertex_type, size_t> update_count_map_type;
   
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

    //! max time
    double max_time_;

    //! the commutative semiring for updates (typically sum_product)
    boost::shared_ptr<commutative_semiring<F> > csr_;

    //! number of updates
    size_t update_count_;

    //! number of edge updates
    size_t edge_update_count_;

    //! Number of splashes
    size_t splash_count_;


    //! number of updates per vertex
    update_count_map_type vertex_update_count_;



  public:

    /**
     * Create an engine (without allocating messages) with the factor
     * graph, splash size, convergence bound, and damping.
     */
    blf_residual_splash_bp(factor_graph_type* factor_graph,
                           size_t splash_size,
                           double bound,
                           double damping, 
			   double max_time = 120.0*60.0) :
      factor_graph_(factor_graph), 
      splash_size_(splash_size), 
      bound_(bound), 
      damping_(damping),
      max_time_(max_time),
      csr_(new sum_product<F>()),
      update_count_(0),
      edge_update_count_(0),
      splash_count_(0) {
      // Ensure that the factor graph is not null
      assert(factor_graph_ != NULL);
    } // end blf_residual_splash_bp


    /**
     * Update the damping value
     */
    void damping(double damping) {
      damping_ = damping;
    }


    /**
     * Reinitialize the messages and then run to convergence.
     */
    bool run() {
      // initialize the messages
      initialize_state();
      // Run splash to convergence
      return splash_to_convergence();
    } // End of run

    

    void initialize_state() {
      // Clear the messages
      messages_.clear();
      // Allocate all messages
      foreach(const vertex_type& u, factor_graph_->vertices()) {        
        foreach(const vertex_type& v, factor_graph_->neighbors(u)) { 
          message_type& msg = message(u,v);
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
      
      // Initialize the counters
      update_count_ = 0;
      edge_update_count_ = 0;
      splash_count_ = 0;
    } // end of initialize
    
    bool splash_to_convergence() {
      assert(schedule_.empty() == false);
      // Start a timer to track progress
      timer ti; ti.start();
      double start_time = ti.current_time();
      double last_time = start_time;
      // While the top element in the schedule is above the bound_
      while(schedule_.top().second > bound_) {
        // Execute a splash
        splash(schedule_.top().first);
        splash_count_++;
	
	// get the current time
	double current_time = ti.current_time();
        // Check the splash count
	if( (current_time - last_time) > 2) {
          std::cout << splash_count_ << ": " 
                    << update_count_ << ": " 
                    << schedule_.top().second << std::endl;
	  last_time = current_time;
        }

	// check if time for early termination is required
	if(current_time - start_time > max_time_) {
	  std::cout << "!!!!!!!!!!! Early Termination !!!!!!!!!" 
		    << std::endl;
	  return false;
	}
      }
      return true;
    } // end of splash_to_convergence


    
    // Construct a splash on a particular vertex
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
      size_t work_in_queue = 0;
      // Set the root to be visited and the first element in the queue
      splash_queue.push_back(root);
      work_in_queue += factor_graph_->num_neighbors(root);
      visited.insert(root);
      // Grow a breath first search tree around the root      
      for(size_t work_in_splash = 0; work_in_splash < splash_size_ 
            && !splash_queue.empty();) {
        // Remove the first element
        vertex_type u = splash_queue.front();
        splash_queue.pop_front();
        size_t work = factor_graph_->num_neighbors(u);
        //        if (work + work_in_splash < splash_size_ || work_in_splash == 0) {
        splash_order.push_back(u);        
        work_in_splash += work;
        work_in_queue -= work;
        //        }
        //        else {
        //          work_in_queue -= work;
        //          continue;
        //        }
        // if we still need more work for the splash
        if(work_in_queue + work_in_splash < splash_size_) {
          // Insert the first element into the tree order If we need
          // more vertices then grow out more Add all the unvisited
          // neighbors to the queue
          foreach(const vertex_type& v, factor_graph_->neighbors(u)) {
            if((visited.count(v) == 0) && schedule_.get(v) > bound_) {
              splash_queue.push_back(v);
              visited.insert(v);
              work_in_queue += factor_graph_->num_neighbors(v);
            }
          } // end of for each neighbors
        }
      } // End of foor loop
    } // End of Generate Splash



    // Update the message changing the destination belief and priority and returning 
    // the belief residual.
    inline std::pair<double,double> update_message(const vertex_type& source,
                                 const vertex_type& target,
                                 const message_type& new_msg) {
      // Get the original message
      message_type& original_msg = message(source,target);

      // Compute the message residual
      double message_residual = norm_1(new_msg, original_msg);
      
      // Make a backup of the old belief
      belief_type prevblf = beliefs_[target];

      // Update the new belief by dividing out the old message and
      // multiplying in the new message.
      belief_type& blf = beliefs_[target];
      blf /= original_msg;
      blf *= new_msg;

      // Ensure that the belief remains normalized
      blf.normalize();

      // Update the message
      original_msg = new_msg;

      // Compute the new residual
      double belief_residual = norm_1(prevblf, blf);
      
      // Sanity checks
      assert(std::isfinite(belief_residual));
      
      // update the priority
      double new_priority = schedule_.get(target) + belief_residual;      

      // For the residual to be finite
      if(!std::isfinite(new_priority)) {
        new_priority = std::numeric_limits<double>::max();
      }

      // update the schedule by adding the delta change in KL
      schedule_.update(target, new_priority);

      return std::make_pair(belief_residual, message_residual);

    } // end of update_message




    inline void send_messages(const vertex_type& source) {
      // Record an additional update count in the aggregate update
      // counts
      vertex_update_count_[source]++;
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
      update_count_++;
    } // end of update messages


    // Recompute the message from the source to target updating the
    // target belief and returning the change in belief value
    inline std::pair<double,double> send_message(const vertex_type& source,
                                                 const vertex_type& target) {
      // here we can assume that the current belief is correct
      belief_type& blf = beliefs_[source];

      // Construct the cavity
      belief_type cavity = blf / message(target, source);

      cavity.normalize();
      // Marginalize out any other variables
      domain_type domain = make_domain(source.is_variable()?
                                       &(source.variable()) :
                                       &(target.variable()));
      message_type new_msg = cavity.marginal(domain);

      // Normalize the message
      new_msg.normalize();
      // Damp messages form factors to variables
      if(target.is_variable()) {
        new_msg = weighted_update(new_msg, 
                                  message(source, target), 
                                  damping_);
      }
      new_msg.normalize();

      // update the message also updating the schedule
      std::pair<double, double> residual =  
        update_message(source, target, new_msg);
 
      // update the update counts
      edge_update_count_++;

      return residual;
    } // end of send_message







    //! get the message from source to target
    inline const message_type& message(const vertex_type& source, 
                                       const vertex_type& target) const {
      // Create iterator typedefs 
      typedef typename message_map_type::const_iterator outer_iterator;
      typedef typename message_map_type::mapped_type::const_iterator 
        inner_iterator;

      outer_iterator outer_iter = messages_.find(source);
      assert(outer_iter != messages_.end());
      inner_iterator inner_iter = outer_iter->second.find(target);
      assert(inner_iter != outer_iter->second.end());

      return messages_[source][target];
    } // end of message

    //! get the message from source to target
    inline message_type& message(const vertex_type& source, 
                                 const vertex_type& target)  {
      return messages_[source][target];
    } // end of message





    /**
     * Compute the belief for a vertex
     */
    const belief_type& belief(variable_type* variable) {
      return beliefs_[factor_graph_->to_vertex(variable)];
    } // end of send_message


    const belief_type& belief(const vertex_type& vert) {
      return beliefs_[vert];
    } // end of send_message


    void belief(std::map<vertex_type, belief_type> &beliefs) {
      beliefs = beliefs_;
    } // end of send_message

    
    //! Compute the map assignment to all variables
    void map_assignment(finite_assignment &mapassg) {
      foreach(const vertex_type &v, factor_graph_->vertices()) {
        if (v.is_variable()) {
          finite_assignment localmapassg = arg_max(beliefs_[v]);
          mapassg[&(v.variable())] = localmapassg[&(v.variable())];
        }
      }
    }


    //! Get the number of updates
    size_t update_count() const { return update_count_; }

    //! Get the number of splash events
    size_t splash_count() const { return splash_count_; }

    //! Get the update count for a particular vertex
    size_t update_count(const vertex_type& vert) const {
      typedef typename update_count_map_type::const_iterator iterator;
      iterator iter = vertex_update_count_.find(vert);
      assert(iter != vertex_update_count_.end());
      return iter->second;
    }


  }; // End of class blf_residual_splash_bp




}; // end of namespace
#include <sill/macros_undef.hpp>



#endif




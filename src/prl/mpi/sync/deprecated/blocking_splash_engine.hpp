#ifndef BLOCKING_SPLASH_ENGINE_HPP
#define BLOCKING_SPLASH_ENGINE_HPP

#include <list>

#include <prl/range/forward_range.hpp>
#include <prl/model/factor_graph_model.hpp>
#include <prl/inference/parallel/basic_update_rule.hpp>
#include <prl/parallel/pthread_tools.hpp>
#include <prl/inference/parallel/message_data.hpp>


// This should be last
#include <prl/macros_def.hpp>

namespace prl {
  
  template<typename F, typename StateManager>
  class blocking_splash_engine {
  public:
    typedef F factor_type;
    typedef typename factor_type::variable_type variable_type;
    typedef typename StateManager::vertex_type vertex_type;    
    typedef typename StateManager::belief_type belief_type;
    typedef basic_update_rule<StateManager> update_rule_type;
    typedef std::vector<vertex_type> ordering_type;

    
  private:
    StateManager* state_;
    size_t splash_size_;
    double bound_;
    double damping_;
    update_rule_type update_rule_;

  public:
    /**
     * This blocking method blocks until the engine finishes and all
     * threads return.  Use the belief(vertex_type) method to access
     * the resulting beliefs after execution. 
     *
     * Damping will typically be set to 0.4 I have removed the default
     * argument because it could produce a compile time bug if the
     * bound (a newly added argument) is not provided.
     *
     */
    blocking_splash_engine() { }

    void initialize(StateManager& state,
                    size_t splash_size,
                    double bound,
                    double damping) {
      state_ = &state;
      splash_size_ = splash_size;
      bound_ = bound;
      damping_ = damping;
      update_rule_ = update_rule_(damping_);
    }


    //! Read the belief from the engine.
    const belief_type& belief(const vertex_type& v) const {
      return state_.belief(v);
    }
    
    // Method launched on each thread
    void blocking_run() {
      int updatecount = 0;
      int iterationcount = 0;
      std::cout << "Starting Run in worker" << std::endl;
      // Preallocate an ordering
      ordering_type order;
      assert(state_ != NULL); 
      while(state_->finished() == false) {
        iterationcount++;
        std::pair<vertex_type, double> top = state_->deschedule_top();
        if (top.second < 0) {
          std::cerr << "No top element available!\n";
          usleep(10000);
          continue;
        }
        vertex_type root = top.first;
        
        // std::cout << root;
        // std::cout << top.second << std::endl;
        
        // Grow a splash ordering
        generate_splash(order, root, splash_size_, bound_);          
        // Push belief from the leaves to the root
        //-------------------------------------------------------------------
        revforeach(const vertex_type& v, order) {
          update_rule_.send_messages(v, *state_);
          updatecount++;
        }
        // Push belief from the root to the leaves (skipping the
        // root which was processed in the previous pass)
        //------------------------------------------------------------------
        foreach(const vertex_type& v, 
                std::make_pair(++order.begin(), order.end())) {
          update_rule_.send_messages(v, *state_);
          updatecount++;
        }
        
        // Update the beliefs separately.  This is done to minimize
        // the number of times beliefs must be updated
        foreach(const vertex_type& v, order) {
          if(v.is_variable()) update_rule_.update_belief(v, *state_);
        }
        
        // enable root to be rescheduled
        state_->schedule(root);
      }
      std::cerr << iterationcount << " iterations" << std::endl;
      std::cerr << updatecount << " updates" << std::endl;
    } // end of run method

    //////////////////////////////////////////////////////////////////////////
    // Helper Functions
  private:    
    // Helper function to fill in an ordering for a splash
    void generate_splash(ordering_type& order, 
                         const vertex_type& root,
                         size_t splash_size,
                         double bound) const {
      typedef std::set<vertex_type> visited_type;
      typedef std::list<vertex_type> queue_type;
      // Create a set to track the vertices visited in the traversal
      visited_type visited;
      queue_type queue;
      // Clear the current ordering
      order.clear();
      // Set the root to be visited and the first element in the queue
      queue.push_back(root);
      visited.insert(root);
      // Grow a breath first search tree around the root      
      for(size_t i = 0; i < splash_size && !queue.empty(); ++i) {
        // Remove the first element
        vertex_type u = queue.front();
        queue.pop_front();
        // Insert the first element into the tree order
        order.push_back(u);
        // If we need more vertices then grow out more
        if(order.size() + queue.size() < splash_size) {
          // Add all the unvisited neighbors to the queue
          foreach(const vertex_type& v, state_->neighbors(u)) {
            double r = state_->residual(v);
            if( (visited.count(v) == 0) &&  r > bound && r >= 0.0 && state_->available(v) ) {      
              queue.push_back(v);
              visited.insert(v);
            }
          } // end of for each neighbors
        } // End of if statement
      } // End of foor loop
    } // End of Generate Splash
  }; // End of class residual splash

} // End of namespace prl

#include <prl/macros_undef.hpp>

#endif 

//End of file



#ifndef SYNC_STATE_MANAGER_HPP
#define SYNC_STATE_MANAGER_HPP

// STL includes
#include <map>
#include <vector>
#include <iostream>




// SILL Includes
#include <sill/model/factor_graph_model.hpp>
#include <sill/parallel/timer.hpp>
#include <sill/factor/util/norms.hpp>
#include <sill/datastructures/mutable_queue.hpp>
#include <sill/mpi/mpi_wrapper.hpp>


// This include should always be last
#include <sill/macros_def.hpp>

/**
 * Defines the async message manager which manages the state of single
 * processor single threaded belief propagation.  This is a very
 * simple state manager which also provides basic %mpi facilities
 */
namespace sill {

  /**
   * Manages the state of a belief propagation algorithm Manages the
   * messages and residuals.
   */
  template <typename F>
  class sync_state_manager {
  public:

    typedef F factor_type;
    typedef factor_type message_type;
    typedef factor_type belief_type;
  
    typedef typename factor_type::result_type    result_type;
    typedef typename factor_type::variable_type  variable_type;
    typedef typename factor_type::domain_type    domain_type;
    typedef typename factor_type::collapse_type  collapse_type;

    typedef factor_graph_model<factor_type> factor_graph_model_type;
    typedef typename factor_graph_model_type::vertex_type vertex_type;

    typedef std::pair<vertex_type, vertex_type> directed_edge;

  private:

    //! type of internal message maps
    typedef std::map<directed_edge, factor_type > message_map_type;

    //! type of belief map
    typedef std::map<vertex_type, factor_type> belief_map_type;
    
    //! the residual priority queue type
    typedef mutable_queue<vertex_type, double> priority_queue_type;
    
    /**
     * The underlying factor graph that this algorithm is solving
     */
    const factor_graph_model_type* factor_graph_;
       
    /**
     * Global lock
     */


    //! Epsilon of error tollerated for convergence
    double epsilon_;
   
    /**
     * The norm used to evaluate the change in messages.  Here we use
     * an L1 norm to measure the change in factors.
     */
    factor_norm_1<message_type> norm_;

    /**
     * The prirority queue which tracks the residual of each vertex
     */
    priority_queue_type schedule_;

    /**
     *  contains the state of convergence for this algorithm
     */
    bool finished_;
  
  public:  

    sync_state_manager(const factor_graph_model_type* factor_graph) :
      factor_graph_(factor_graph), finished_(false) {

      assert(factor_graph_ != NULL);
      assert(po_ != NULL);
      // Initialize the priority queue
      foreach(const vertex_type& v, ownership_) 
        schedule_.enqueue(v, std::numeric_limits<double>::max_value());
      // Initialize the ownerhsip set
      foreach(const vertex_type& v, ownership_) 
        ownership_.insert(v);
    } // end of constructor

    /**
     * evaluates whether the particular vertex is local
     */
    bool is_local(const vertex_type& vertex) {
      return ownership_.find(vertex) != ownership_.end();
    } 

  
    /**
     * Returns all the factors associated with a particular variable
     * except unary factors associated with that variable.
     */
    forward_range<const vertex_type&> neighbors(const vertex_type& v) const {
      return factor_graph_->neighbors(v);
    }

    /**
     * This method checks out the messages from the vertex v1 to the
     * vertex v2. 
     */
    message_type* checkout(const vertex_type& v1,
                           const vertex_type& v2,
                           const ReadWrite& rw){
      // if this is for writing then we give the buffer
      if(rw = Writing) {
        // the buffer cannot currently be in use
        assert(buffer_in_use_ == false);
        buffer_in_use_ = true;
        message_buffer_ = 1;
        return &message_buffer_;
      } else {
        // otherwise return the message in the map
        return &(messages_[directed_edge(v1,v2)]);
      }
    }

  
    /**
     * Same as checkout except without assertion
     */
    message_type* try_checkout(const vertex_type& v1,
                               const vertex_type& v2,
                               const ReadWrite& rw){
      // if this is for writing then we try and give the buffer
      if(rw = Writing) {
        // if the buffer is in use return a NULL
        if(buffer_in_use_){
          return NULL;
        } else {
          buffer_in_use_ = true;
          message_buffer_ = 1;
          return &message_buffer_;          
        }
      } else {
        // find f in the map
        return &(messages_[directed_edge(v1,v2)]);
      }
    }
    
    /**
     * compute the belief associated with a vertex
     */
    belief_type* checkout_belief(const vertex_type& v){
      return &beliefs_[v];
    }
    
    /**
     * just update the residual and the priority
     */
    void checkin(const vertex_type& v1, 
                 const vertex_type& v2,
                 const message_type* msg) {
      // Cannot checkin a null message
      assert(msg != NULL);
      // First test to see if this was a write checkout
      if(msg == &message_buffer_) {
        // Get the current message
        message_type& stored_msg_ref(messages_[directed_edge(v1,v2)]);
        // if we own the receiving vertex then we need to compute the residual
        // and update the necessary scheduling info
        if(ownership_.find(v2) != ownership_.end()) {
          // this is a write checkin
          double residul = norm_(message_buffer_, stored_msg);
          // if necessary increase the priority of the vertex
          // this is effectively a promote
          if(schedule_[v2] < residual) schedule_.update(v2, residual);
        } 
        // assing the message_buffer
        stored_msg = message_buffer_;
        // The buffer is no longer in use
        buffer_in_use_ = false;
      } else { 
        // This is a read checkin so we do nothing
        // NOP
      }
    }

    /**
     * required for compatibitlity
     */
    void checkin_belief(const vertex_type& v, belief_type* b){
      // NOP 
    }
    
  
    /**
     * Gets the top factor and deactivates it from the scheduling
     * queue.  This vertex will not be accessible by deschedule_top
     * until schedule(v) is invoked on that vertex.
     */
    std::pair<vertex_type, double> deschedule_top() {
      return schedule_.top();
    }

    /**
     * required for compatiblity
     */
    void schedule(vertex_type v) {
      // NOP 
    }

    /**
     * This is function should set the priority / residual of the
     * vertex to 0
     */
    void mark_visited(vertex_type v) {
      schedule_.update(v, 0.0);
    }

    
  
    //! Get the residual for the vertex
    double residual(const vertex_type& v) {
      return schedule_[v];
    }
  
    //! Gets the variable residual  
    double residual(const variable_type v) {
      return residual(vertex_type(v));
    }
  
    //! Gets the factor residual
    double residual(const factor_type* f) {
      return residual(vertex_type(f));
    }

    //! Gets the termination bound criteria
    inline double get_bound(){
      return epsilon_;
    }

    /**
     * Determines whether the state of the execution is finished. This
     * will return true if the highest residual message has a residual
     * less than epsilon initialized at construction
     */
    bool finished() {
      return finished_;
    }


    /**
     * This function should:
     *  1) Check to see if another process is transmitting to us
     *  2) Initiate passive reception of messages
     *  3) Initiate passive transmission of any messages
     *  4) Check pending receptions
     *  5) Check pending transmissions
     *  6) Update schedule
     *  7) Check termination
     */
    void mpi_synch() {
      std::cout << "Function incomplete!!" << std::endl;
    }

  }; // End of async message manager


} // End of namespace

#include <sill/macros_undef.hpp>

#endif

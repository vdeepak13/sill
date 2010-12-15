#ifndef BASIC_UPDATE_RULE_HPP
#define BASIC_UPDATE_RULE_HPP

#include <limits>

#include <prl/model/factor_graph_model.hpp>
#include <prl/math/gdl_enum.hpp>
#include <prl/inference/parallel/basic_state_manager.hpp>
#include <prl/macros_def.hpp>

namespace prl {

  /**
   * The basic update rule satisfies the update rule concept providing
   * the opeations:
   *   - send_messages(vertex)
   *   - update_belief(vertex)
   *
   * The basic update rule requires a commutative semiring of either
   *   - sum -- product
   *   - max -- product
   * 
   */
  template<typename StateManager>
  class basic_update_rule {

    // Define the basic types
    typedef typename StateManager::factor_type    factor_type;
    typedef typename factor_type::domain_type     domain_type;
    typedef typename factor_type::variable_type   variable_type;
    typedef typename StateManager::belief_type    belief_type;
    typedef typename StateManager::message_type   message_type;
    typedef typename StateManager::vertex_type    vertex_type;

    //! The ammount of damping in the update
    double damping_;

    //! the commutative semiring for updates (typically sum_product)
    commutative_semiring csr_;

  public:
   
    /**
     * Initialize a basic update rule providing damping and the associated 
     * commutative semiring.
     */
    basic_update_rule(double damping = 0.3,
                      commutative_semiring csr = sum_product) :
      damping_(damping), csr_(csr) { }

//     /**
//      * Update a vertex.  Recall that a vertex in a factor graph is
//      * either a variable or a factor.  Therefore this message first
//      * determines whether the vertex corresponds to a variable or
//      * factor and then invokes the appropriate send messages routine.
//      * 
//      * Update then receives all inbound messages and computes all
//      * outbound messages.  In addition if the vertex input corresponds
//      * to a variable this method also computes the belief for that
//      * variable.
//      */
//     inline void update(const vertex_type& vertex,
//                        StateManager& state) {
//       send_messages(vertex, state);
//       if( vertex.is_variable() ) {
//         update_belief(vertex, state);
//       }
//       // Mark the vertex as visited (this sets inbound residual to 0)
//       state.mark_visited(vertex);
//     }

    /**
     * Receive all messages into the factor f and compute all new
     * outbound messages.
     */
    inline void send_messages(const vertex_type& source,
                              StateManager& state) {
      // For each of the neighboring vertices (vertex_target) of
      // vertex compute the message from vertex to vertex_target
      foreach(const vertex_type& dest, state.neighbors(source)) {
        // Checkout the destination message
        send_message(source, dest, state);
      }
      // Mark the vertex as having been visited
      state.mark_visited(source);
    } // end of update messages


    /**
     * Send the message from vertex_source to vertex_target.  Note
     * that if another processor is currently trying to send this
     * message then this routine will simply return;
     */
    inline void send_message(const vertex_type& vertex_source,
                             const vertex_type& vertex_target,
                             StateManager& state) { 
      // Try to get the message from the state manager
      message_type* msg = state.try_checkout(vertex_source, 
                                             vertex_target, 
                                             Writing);
      // If the message is NULL then we cannot proceed so return
      if(msg == NULL)  return;
      // Otherwise we may proceed
      assert(msg != NULL);
      // Backup the message so extra computation may be done at the end
      message_type msg_backup = *msg;

      // Verify that the message is nonzero
      // foreach(const double value, msg->table()) { assert(value > 0); }

      // Depending on if the vertex is a variable or a factor we do
      // something different
      if(vertex_source.is_variable()) {
        // here we assume that we can initialize the factor to a
        // constant distribution.  This may change the size of the
        // message and require extra allocations?
        *msg = message_type(msg->arguments(), 1).normalize();
      } else if(vertex_source.is_factor()) {
        // Set the message equal to the factor.  This will increase
        // the size of the message and require an allocation.
        *msg = vertex_source.factor();
      } else {
        // The vertex must either correspond to a variable or a factor
        // This state should not be reachable
        assert(false);
      }

      // For each of the neighbors of the vertex
      foreach(const vertex_type& vertex_other, 
              state.neighbors(vertex_source)) {
        // if this is not the dest_v
        if(vertex_other != vertex_target) {
          // Checkout a copy of the message for reading only
          message_type* in_msg = 
            state.try_checkout(vertex_other, vertex_source, Reading);
          if(in_msg != NULL) {
            // Combine the in_msg with the destination factor
            msg->combine_in(*in_msg, csr_.dot_op);
            // Check in the message (no longer needed)
            state.checkin(vertex_other, vertex_source, in_msg);
            // Here we normalize after each iteration for numerical
            // stability.  This could be very costly for large factors.
            msg->normalize();
          }
        }
      }        
      // If this is a message from a factor to a variable then we
      // must marginalize out all variables except the the target
      // variable.  
      if(vertex_target.is_variable()) {
        (*msg) = msg->collapse(csr_.cross_op, 
                                make_domain(&vertex_target.variable()));
      }

      
      // Normalize the message
      msg->normalize();

      // Save  the damped message to the destination message
      // lets only damp the factor to variable updates
      // otherwise we will 'doubly damp' the messages in a pairwise MRF
      if(vertex_target.is_variable()) {
        *msg = weighted_update(*msg, msg_backup, damping_);
      }
      
      // checkin the final message to the state manager
      state.checkin(vertex_source, vertex_target, msg);

    } // end of send_message


    /**
     * Receive all messages and compute the new belief. 
     */
    inline void update_belief(const vertex_type& vertex,
                              StateManager& state) {
      
      // Get the belief from the state manager
      belief_type* blf = state.checkout_belief(vertex);
      // Wipe out the old value for the belief
      if(vertex.is_variable()) {
        *blf = message_type(blf->arguments(), 1).normalize();
      } else if(vertex.is_factor()) {
        *blf = vertex.factor();
      } else {
        assert(false);
      }
      // For each of the neighbor variables 
      foreach(const vertex_type& vertex_source, state.neighbors(vertex)) {
        // get the in message
        message_type* in_msg = 
          state.try_checkout(vertex_source, vertex, Reading);
        if(in_msg != NULL) {
          // Combine the in_msg with the destination factor
          blf->combine_in(*in_msg, csr_.dot_op);
          // return the message to the state manager
          state.checkin(vertex_source, vertex, in_msg);
          // normalize the belief
          blf->normalize();
        }
      }  
      // Do an extra normalization (just in case no messages were
      // available)
      blf->normalize();
      // ASSERT WE BLF IS A VALID DISTRIBUTION (we should check this)
      // Save the belief
      state.checkin_belief(vertex, blf);
    }// End of update belief    
  }; // End of basic update rule
} // end of namespace



#include <prl/macros_undef.hpp>
#endif







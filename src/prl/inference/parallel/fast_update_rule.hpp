#ifndef FAST_UPDATE_RULE_HPP
#define FAST_UPDATE_RULE_HPP

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
   * This is the fast update rule which is only safe to use in a
   * single threaded context
   */
  template<typename StateManager>
  class fast_update_rule {

    // Define the basic types
    typedef typename StateManager::factor_type    factor_type;
    typedef typename factor_type::domain_type     domain_type;
    typedef typename factor_type::variable_type   variable_type;
    typedef typename StateManager::belief_type    belief_type;
    typedef typename StateManager::message_type   message_type;
    typedef typename StateManager::vertex_type    vertex_type;

    //! The ammount of damping in the update
    double damping_;
    int b;
    //! the commutative semiring for updates (typically sum_product)
    commutative_semiring csr_;

  public:
   
    /**
     * Initialize a basic update rule providing damping and the associated 
     * commutative semiring.
     */
    fast_update_rule(double damping = 0.3,
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
    void send_messages(const vertex_type& vertex,
                              StateManager& state) {
      // Get the belief from the state manager
      belief_type* blf;
      // Wipe out the old value for the belief
      if(vertex.is_variable()) {
        blf = state.checkout_belief(vertex);
        (*blf) = 1;
        blf->normalize();
//        std::cout << "var\n";
      } else if(vertex.is_factor()) {
        blf = new belief_type();
        (*blf) = vertex.factor();
        blf->normalize();
//        std::cout << "fact\n";
      } else {
        assert(false);
      }
//      std::cout << blf->arguments()<<"\n";
      std::vector<vertex_type> neighbors;
      std::vector<message_type*> neighbor_inmsg;
      // For each of the neighbor variables 
//      std::cout << (*blf);
      foreach(const vertex_type& vertex_source, state.neighbors(vertex)) {
        // get the in message
        message_type* in_msg = 
          state.checkout(vertex_source, vertex, Reading);
          // remember the messages and which neighbor it came from
          neighbors.push_back(vertex_source);
          neighbor_inmsg.push_back(in_msg);
          // Combine the in_msg with the destination factor
          if (vertex.is_variable()) {
            assert(in_msg->arguments().size() == 1);
            assert(in_msg->arguments().contains(&(vertex.variable())));
          }
          else if (vertex.is_factor()) {
            assert(blf->arguments().contains(&(vertex_source.variable())));
          }
          blf->combine_in(*in_msg, csr_.dot_op);
//          std::cout << "Include: " << (*in_msg);
          // normalize the belief
          blf->normalize();
//          std::cout << (*blf);
      }  
      //std::cout<< "------------------------------------";
      // compute the outgoing messages
      message_type oldmessage;
      for (size_t i = 0; i< neighbors.size(); ++i) {
        message_type* out_msg = state.checkout(vertex, neighbors[i], Writing);
        // divide out the incoming message here
        oldmessage = (*out_msg);
//        std::cout << "Exclude: " << (*neighbor_inmsg[i]);
        if (neighbors[i].is_variable()) {
          (*out_msg) = blf->collapse(make_domain(&neighbors[i].variable()), csr_.cross_op);
          out_msg->combine_in(*(neighbor_inmsg[i]), divides_op);
        }
        else {
          (*out_msg) = combine(*blf, *(neighbor_inmsg[i]), divides_op);
        }
        
        out_msg->normalize();
//        std::cout<<*out_msg;
        // Save  the damped message to the destination message
        // lets only damp the factor to variable updates
        // otherwise we will 'doubly damp' the messages in a pairwise MRF
        if(neighbors[i].is_variable()) {
          (*out_msg) = weighted_update(*out_msg, oldmessage, damping_);
        }
        // checkin the final message to the state manager
        state.checkin(neighbors[i], vertex, neighbor_inmsg[i]);
        state.checkin(vertex, neighbors[i], out_msg);
      }
//      getchar();
      // lets checkin the belief while we are at it
      if (vertex.is_variable()) state.checkin_belief(vertex,blf);
      else delete blf;
      state.mark_visited(vertex);
    } 

    /**
     * Receive all messages and compute the new belief. 
     */
    inline void update_belief(const vertex_type& vertex,
                              StateManager& state) {
      
      // Get the belief from the state manager
      belief_type* blf = state.checkout_belief(vertex);
      // Wipe out the old value for the belief
      if(vertex.is_variable()) {
        *blf = 1;
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







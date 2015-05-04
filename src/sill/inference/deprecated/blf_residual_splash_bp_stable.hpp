#ifdef BLF_RESIDUAL_SPLASH_BP_HPP
#error "Do not include both blf_residual_splash_bp.hpp and blf_residual_splash_bp_stable.hpp"
#endif
#ifndef BLF_RESIDUAL_SPLASH_BP_STABLE_HPP
#define BLF_RESIDUAL_SPLASH_BP_STABLE_HPP


// STL includes
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>

// SILL Includes
#include <sill/model/factor_graph_model.hpp>
#include <sill/factor/util/norms.hpp>
#include <sill/math/gdl_enum.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/datastructure/mutable_queue.hpp>
#include <sill/parallel/timer.hpp>
#include <sill/inference/loopy/bp_convergence_measures.hpp>


// This include should always be last
#include <sill/macros_def.hpp>
namespace sill {

  template<typename F>
  class blf_residual_splash_bp {
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

    //  typedef factor_norm_1<message_type> norm_type;
    typedef factor_norm_inf_log<message_type> norm_type;


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
//    double bound_;

    //! level of damping 1.0 is fully damping and 0.0 is no damping
    double damping_;

    //! the commutative semiring for updates (typically sum_product)
    commutative_semiring csr_;

    //! this object tells us when we're done
    residual_splash_convergence_measure* convergence_indicator_;

    //! true if convergence_indicator_ was allocated by this object and
    //! must be deleted by this object.
    //! need this to support the old constructor interface.
    //! TODO: phase out
    bool own_convergence_indicator_;


  public:
    map<vertex_type,int> degreeupdatecount;



    blf_residual_splash_bp(factor_graph_type* factor_graph,
                           size_t splash_size,
                           double bound,
                           double damping,
                           commutative_semiring csr = sum_product) :
      factor_graph_(factor_graph),
      splash_size_(splash_size),
      damping_(damping),
      csr_(csr){
      // Ensure that the factor graph is not null
      assert(factor_graph_ != NULL);

      convergence_indicator_ = new residual_splash_convergence_measure(bound);
      own_convergence_indicator_ = true;

    } // end blf_residual_splash_bp

    blf_residual_splash_bp(factor_graph_type* factor_graph,
                           size_t splash_size,
                           residual_splash_convergence_measure* convergence_indicator,
                           double damping,
                           commutative_semiring csr = sum_product) :
      factor_graph_(factor_graph),
      splash_size_(splash_size),
      damping_(damping),
      csr_(csr),
      convergence_indicator_(convergence_indicator),
      own_convergence_indicator_(false){
      // Ensure that the factor graph is not null
      assert(factor_graph_ != NULL);
    } // end blf_residual_splash_bp

    void run(int &spcount, int &upcount) {
      // initialize the messages
      initialize_state();
      // Ran splash to convergence
      spcount = splash_to_convergence();
      // Close the file stream
      upcount = update_count;

    } // End of run


    void initialize_state() {
      norm_type norm;
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



    size_t update_count;

    int splash_to_convergence() {
      timer ti;
      ti.start();
      double outtime = ti.current_time();
      assert(schedule_.size() > 0);
      size_t splash_count = 0;
      update_count = 0;
      convergence_indicator_->start();
      while(!convergence_indicator_->is_converged(schedule_.top().second, splash_count)) {
        splash(schedule_.top().first);
        splash_count++;
        if(ti.current_time() - outtime > 2) {
          std::cout << splash_count << ": " << update_count<<": " << schedule_.top().second << std::endl;
          outtime = ti.current_time();
        }
      }
      return splash_count;
    } // end of splash_to_convergence



    void splash(const vertex_type& root) {
      ordering_type splash_order;
      // Grow a splash ordering
      generate_splash(root, splash_order);
      // Push belief from the leaves to the root
      revforeach(const vertex_type& v, splash_order) {
        send_messages(v);
        update_count++;
      }
      // Push belief from the root to the leaves (skipping the
      // root which was processed in the previous pass)
      foreach(const vertex_type& v,
              std::make_pair(++splash_order.begin(), splash_order.end())) {
        send_messages(v);
        update_count++;
      }
    } // End of splash_once



//     void generate_splash(const vertex_type& root,
//                          ordering_type& splash_order) {
//       typedef std::set<vertex_type> visited_type;
//       typedef std::list<vertex_type> queue_type;
//       // Create a set to track the vertices visited in the traversal
//       visited_type visited;
//       queue_type splash_queue;
//       // Set the root to be visited and the first element in the queue
//       splash_queue.push_back(root);
//       visited.insert(root);
//       // Grow a breath first search tree around the root
//       for(size_t i = 0; i < splash_size_ && !splash_queue.empty(); ++i) {
//         // Remove the first element
//         vertex_type u = splash_queue.front();
//         splash_queue.pop_front();
//         // Insert the first element into the tree order
//         splash_order.push_back(u);
//         // If we need more vertices then grow out more
//         if(splash_order.size() + splash_queue.size() < splash_size_) {
//           // Add all the unvisited neighbors to the queue
//           foreach(const vertex_type& v, factor_graph_->neighbors(u)) {
//             if((visited.count(v) == 0) && schedule_.get(v) > bound_) {
//               splash_queue.push_back(v);
//               visited.insert(v);
//             }
//           } // end of for each neighbors
//         } // End of if statement
//       } // End of foor loop
//     } // End of Generate Splash




    /**
     * This function computes the splash ordering (a BFS) search for
     * the root vertex
     */

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
        if (work + work_in_splash < splash_size_ || work_in_splash == 0) {
          splash_order.push_back(u);
          work_in_splash += work;
          work_in_queue -= work;
        }
        else {
          work_in_queue -= work;
          continue;
        }
        // if we still need more work for the splash
        if(work_in_queue + work_in_splash < splash_size_) {
          // Insert the first element into the tree order If we need
          // more vertices then grow out more Add all the unvisited
          // neighbors to the queue
          foreach(const vertex_type& v, factor_graph_->neighbors(u)) {
            if((visited.count(v) == 0) && schedule_.get(v) > convergence_indicator_->residual_bound()) {
              splash_queue.push_back(v);
              visited.insert(v);
              work_in_queue += factor_graph_->num_neighbors(v);
            }
          } // end of for each neighbors
        }
      } // End of foor loop
    } // End of Generate Splash



    inline void update_message(const vertex_type& source,
                               const vertex_type& target,
                               const message_type& new_msg) {
       //     assert(new_msg.minimum() > 0.0);
      // Create a norm object (this should be lightweight)
      factor_norm_1<message_type> norm;
      // Get the original message
      message_type& original_msg = messages_[source][target];
      belief_type prevblf = beliefs_[target];
      belief_type& blf = beliefs_[target];
      blf.combine_in(original_msg, divides_op);
      blf.combine_in(new_msg, csr_.dot_op);
      blf.normalize();
      //      assert(blf.minimum() > 0.0);
      original_msg = new_msg;
      //double new_residual = norm(blf, last_beliefs_[target]);
      //double new_residual = blf.entropy() * norm(blf, last_beliefs_[target]);
      /*double new_residual = blf.relative_entropy(last_beliefs_[target]) +
                            last_beliefs_[target].relative_entropy(blf);*/
      //double new_residual = blf.relative_entropy(last_beliefs_[target]);
      //double new_residual = last_beliefs_[target].relative_entropy(blf);
/*            double new_residual = schedule_.get(target) + prevblf.relative_entropy(blf)
                            - last_beliefs_[target].relative_entropy(prevblf)
                             + last_beliefs_[target].relative_entropy(blf); */
      // cross entropy(p,q) = KL(p,q) + entropy(p)


      double new_residual = schedule_.get(target);
      if (target.is_factor()) {
        new_residual += fabs(prevblf.entropy() - blf.entropy() -
                             (blf.relative_entropy(target.factor()) + blf.entropy())
                             + (prevblf.relative_entropy(target.factor()) + prevblf.entropy()));
      }
      else {
        new_residual +=(factor_graph_->num_neighbors(target) - 1) *fabs(blf.entropy() - prevblf.entropy());
       // new_residual +=fabs(blf.entropy() - prevblf.entropy());
      }
      /*
      double new_residual = schedule_.get(target) +
            (factor_graph_->num_neighbors(target) - 1)*std::fabs(blf.relative_entropy(prevblf) +blf.entropy() - prevblf.entropy());
      */
      // not sure if this is necessary now
      if (std::isnan(new_residual)) new_residual = std::numeric_limits<double>::infinity();

      /*std::cout << blf << "\n";
      std::cout << last_beliefs_[target] << "\n";
      std::cout << new_residual << "\n";
      getchar();*/
      // Update the residual
      //if(schedule_.get(target) < new_residual )
        schedule_.update(target, new_residual);
    } // end of update_message




    inline void send_messages(const vertex_type& source) {
        degreeupdatecount[source]++;
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
      cavity.normalize();
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
      new_msg.normalize();
      // update the message also updating the schedule
      update_message(source, target, new_msg);
    } // end of send_message



//     inline void send_message(const vertex_type& source,
//                              const vertex_type& target) {
//       message_type new_msg;
//       if(source.is_variable()) {
//         // create a temporary message to store the result into
//         domain_type domain = make_domain(source.is_variable() ?
//                                          &(source.variable()) :
//                                          &(target.variable()));
//         new_msg = message_type(domain, 1.0).normalize();
//         // If the source was a factor we multiply in the factor potential
//       } else {
//         // Set the message equal to the factor.  This will increase
//         // the size of the message and require an allocation.
//         new_msg = source.factor();
//       }
//       // For each of the neighbors of the vertex
//       foreach(const vertex_type& other,
//               factor_graph_->neighbors(source)) {
//         // if this is not the dest_v
//         if(other != target) {
//           // Combine the in_msg with the destination factor
//           new_msg.combine_in( messages_[other][source], csr_.dot_op);
//           // Here we normalize after each iteration for numerical
//           // stability.  This could be very costly for large factors.
//           new_msg.normalize();
//         }
//       }
//       // If this is a message from a factor to a variable then we
//       // must marginalize out all variables except the the target
//       // variable.
//       if(source.is_factor()) {
//         new_msg = new_msg.collapse(make_domain(&target.variable()), csr_.cross_op);
//       }
//       // Normalize the message
//       new_msg.normalize();
//       // Damp messages form factors to variables
//       if(target.is_variable()) {
//         new_msg = weighted_update(new_msg,
//                                   messages_[source][target],
//                                   damping_);
//       }
//       // update the message also updating the schedule
//       update_message(source, target, new_msg);
//     } // end of send_message


    /**
     * Compute the belief for a vertex
     */
    const belief_type& belief(variable_type variable) {
      return beliefs_[vertex_type(variable)];
    } // end of send_message

    void belief(std::map<vertex_type, belief_type> &beliefs) {
      beliefs = beliefs_;
    } // end of send_message

    void get_map_assignment(finite_assignment &mapassg) {
      foreach(const vertex_type &v, factor_graph_->vertices()) {
        if (v.is_variable()) {
          finite_assignment localmapassg = arg_max(beliefs_[v]);
          mapassg[&(v.variable())] = localmapassg[&(v.variable())];
        }
      }
    }

  }; // End of class blf_residual_splash_bp




}; // end of namespace
#include <sill/macros_undef.hpp>



#endif

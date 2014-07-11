#ifndef ROUND_ROBIN_ENGINE
#define ROUND_ROBIN_ENGINE


#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>

#include <sill/model/factor_graph_model.hpp>
#include <sill/factor/norms.hpp>
#include <sill/factor/commutative_semiring.hpp>
#include <sill/datastructure/mutable_queue.hpp>

// This include should always be last
#include <sill/macros_def.hpp>

namespace sill {

  template<typename F>
  class round_robin_bp {
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

    //! output stream code
    std::ofstream updates_out_;
    std::ofstream likelihood_out_;

    //! number of splashes
    size_t rounds_;
    
    //! number of updates
    size_t update_count_;

  public:
    
    std::map<vertex_type, size_t> degreeupdatecount;



    round_robin_bp(factor_graph_type* factor_graph,
                   size_t splash_size,
                   double bound,
                   double damping) :
      factor_graph_(factor_graph), 
      splash_size_(splash_size), 
      bound_(bound), 
      damping_(damping),
      updates_out_("updates.txt"),
      likelihood_out_("likeli.txt"),
      rounds_(0),
      update_count_(0) {
      // Ensure that the factor graph is not null
      assert(factor_graph_ != NULL);
    } // end round_robin_bp

    size_t rounds() const { return rounds_; }
    size_t update_count() const { return update_count_; }

    void run() {
      // initialize the messages
      initialize_state();
      // Ran splash to convergence
      round_robin_to_convergence();
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
      rounds_ = 0;
      update_count_ = 0;
    } // end of initialize

    
    void round_robin_to_convergence() {
      timer ti;
      ti.start();
      double starttime = ti.current_time();
      double outtime = ti.current_time();

      assert(schedule_.size() > 0);
      update_count_ = 0;
      while(schedule_.top().second > bound_) {
        roundrobin();
        rounds_++;
        std::cout << rounds_ << ": " << update_count_ 
                  << ": " << schedule_.top().second << std::endl;
        
        if(ti.current_time() - outtime > 2) {
          finite_assignment f;
          map_assignment(f);
          likelihood_out_.precision(10);

          likelihood_out_ << update_count_ << ", " 
                          << factor_graph_->log_likelihood(f) 
                          << std::endl;
          outtime = ti.current_time();
        }

        if (ti.current_time() - starttime > 1000) break;
      }
    } 

    void roundrobin() {
      // Push belief from the leaves to the root
      foreach(const vertex_type& v, factor_graph_->vertices()) {
        send_messages(v);
        update_count_++;
      }
    } // End of roundrobin

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
      blf /= original_msg;
      blf *= new_msg;
      blf.normalize();
      //      assert(blf.minimum() > 0.0);
      original_msg = new_msg;
      
      
      double new_residual = schedule_.get(target);
      if (target.is_factor()) {
        new_residual += fabs(prevblf.entropy() - blf.entropy() - 
                             (blf.relative_entropy(target.factor()) + 
                              blf.entropy()) 
                             + (prevblf.relative_entropy(target.factor()) + 
                                prevblf.entropy()));
      } else { 
        new_residual += (factor_graph_->num_neighbors(target) - 1) * 
          fabs(blf.entropy() - prevblf.entropy());
       // new_residual +=fabs(blf.entropy() - prevblf.entropy());
      }
      /*
      double new_residual = schedule_.get(target) +
      (factor_graph_->num_neighbors(target) -
      1)*std::fabs(blf.relative_entropy(prevblf) +blf.entropy() -
      prevblf.entropy());
      */
      // not sure if this is necessary now
      if (std::isnan(new_residual)) 
        new_residual = std::numeric_limits<double>::infinity();

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
      updates_out_ << factor_graph_->vertex2id(source) << std::endl;
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
      belief_type cavity = blf / messages_[target][source];
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
                                  messages_[source][target], 
                                  damping_);
      }
      new_msg.normalize();
      // update the message also updating the schedule
      update_message(source, target, new_msg);
    } // end of send_message




    /**
     * Compute the belief for a vertex
     */
    const belief_type& belief(variable_type* variable) {
      return beliefs_[vertex_type(variable)];
    } // end of send_message

    void belief(std::map<vertex_type, belief_type> &beliefs) {
      beliefs = beliefs_;
    } // end of send_message
    
    void map_assignment(finite_assignment &mapassg) {
      foreach(const vertex_type &v, factor_graph_->vertices()) {
        if (v.is_variable()) {
          finite_assignment localmapassg = arg_max(beliefs_[v]);
          mapassg[&(v.variable())] = localmapassg[&(v.variable())];
        }
      }
    }

  }; // End of class round_robin_bp

} // end of namespace

#include <sill/macros_undef.hpp>

#endif

#ifdef ROUND_ROBIN_ENGINE
#error "Do not include both round_robin_bp.hpp and round_robin_bp_stable.hpp"
#endif
#ifndef ROUND_ROBIN_ENGINE_STABLE
#define ROUND_ROBIN_ENGINE_STABLE


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
#include <sill/inference/loopy/bp_convergence_measures.hpp>


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



    round_robin_bp(factor_graph_type* factor_graph,
                           size_t splash_size,
                           double bound,
                           double damping,
                           commutative_semiring csr = sum_product) :
      factor_graph_(factor_graph),
      splash_size_(splash_size),
      damping_(damping),
      csr_(csr) {

      convergence_indicator_ = new residual_splash_convergence_measure(bound);
      own_convergence_indicator_ = true;

      // Ensure that the factor graph is not null
      assert(factor_graph_ != NULL);
    } // end round_robin_bp

    round_robin_bp(factor_graph_type* factor_graph,
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

    void run(int &extiter, int &upcount) {
      // initialize the messages
      initialize_state();
      // Ran splash to convergence
      extiter = round_robin_to_convergence();
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

    int round_robin_to_convergence() {
      assert(schedule_.size() > 0);
      size_t extiter_count = 0;
      update_count = 0;
      convergence_indicator_->start();
      while(!convergence_indicator_->is_converged(schedule_.top().second, update_count) ) {
        roundrobin();
        extiter_count++;
        std::cout << extiter_count << ": " << update_count<<": " << schedule_.top().second << std::endl;

      }
      return extiter_count;
    }

    void roundrobin() {
      // Push belief from the leaves to the root
      foreach(const vertex_type& v, factor_graph_->vertices()) {
        send_messages(v);
        update_count++;
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
      message_type new_msg = cavity.collapse(csr_.cross_op, domain);
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

  }; // End of class round_robin_bp




}; // end of namespace
#include <sill/macros_undef.hpp>



#endif

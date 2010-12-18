#ifndef RESIDUAL_SPLASH_BP_HPP
#define RESIDUAL_SPLASH_BP_HPP


// STL includes
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <sstream>
#include <limits>

// PRL Includes
#include <sill/model/factor_graph_model.hpp>
#include <sill/factor/norms.hpp>
#include <sill/math/gdl_enum.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/datastructure/mutable_queue.hpp>

// boost timer include needed for cpu time convergence threshold
// factor it out to other file in the future?
#include <boost/timer.hpp>

// This include should always be last
#include <sill/macros_def.hpp>
namespace sill {

  //! interface of classes that determine convergence
  class residual_splash_convergence_measure{
  private:
    //! residual convergence bound
    const double residual_bound_;

    //! determines if start() has been called, need to enforce it for timers
    bool started_;
    
  public:

    residual_splash_convergence_measure(double residual_bound) :
      residual_bound_(residual_bound), started_(false){
      assert(residual_bound_ > 0);
    }

    //! returns true iff the algorithm converged after this update
    virtual bool is_converged( double max_residual,
                                  size_t last_iteration_updates){
      assert(started_);

      if(max_residual < residual_bound_)
        return true;
      else return false;
    }

    double residual_bound(){ return residual_bound_; };

    //! use this to initialize timers in the measures that use them
    virtual void start(){
      assert(!started_);
      started_ = true;
    };

    virtual ~residual_splash_convergence_measure(){};
  };


  class updates_count_convergence_measure :
    public residual_splash_convergence_measure {

  private:
    //! max updates allowed. negative means unlimited
    int max_updates_, total_updates_;

  public:
    updates_count_convergence_measure(double residual_bound, int max_updates) :
      residual_splash_convergence_measure(residual_bound),
      max_updates_(max_updates), total_updates_(0){};

    virtual bool is_converged( double max_residual,
                                  size_t last_iteration_updates){

      if(residual_splash_convergence_measure::is_converged(
            max_residual, last_iteration_updates))
        return true;

      total_updates_ += last_iteration_updates;
      if( max_updates_ > 0 && total_updates_ > max_updates_)
        return true;
      else return false;
    }
  };

  class cpu_time_convergence_measure :
    public residual_splash_convergence_measure {

  private:
    //! residual convergence bound
    const double time_bound_;

    //! cpu timer
    boost::timer* timer_;

  public:

    cpu_time_convergence_measure(double residual_bound, double time_bound) :
      residual_splash_convergence_measure(residual_bound),
      time_bound_(time_bound), timer_(NULL){
      assert(time_bound_ > 0);
    }

    virtual void start(){
      residual_splash_convergence_measure::start();
      //restart the timer
      timer_ = new boost::timer();
    }

    virtual bool is_converged( double max_residual,
                                  size_t last_iteration_updates){
      if(residual_splash_convergence_measure::is_converged(
            max_residual, last_iteration_updates))
        return true;

      else if( timer_->elapsed() > time_bound_)
        return true;
      else return false;
    }

    virtual ~cpu_time_convergence_measure(){
      if(timer_ != NULL)
        delete timer_;
    }
  };


  template<typename F>
  class residual_splash_bp {
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


    ///////////////////////////////////////////////////////////////////////
    // Data members
  private:
    //! pointer to the factor graph
    factor_graph_type* factor_graph_;

    //! the schedule of vertices
    schedule_type schedule_;

    //! messages
    message_map_type messages_;

    //! the size of a splash
    size_t splash_size_;

    //! level of damping 1.0 is fully damping and 0.0 is no damping
    double damping_;

    //! the commutative semiring for updates (typically sum_product)
    commutative_semiring csr_;

      //! output stream code
    //! maximum allowed number of splash updates. negative value means unlimited
    int max_updates_;

    //! this object tells us when we're done
    residual_splash_convergence_measure* convergence_indicator_;

    //! true if convergence_indicator_ was allocated by this object and
    //! must be deleted by this object.
    //! need this to support the old constructor interface.
    //! TODO: phase out
    bool own_convergence_indicator_;

    std::ofstream updates_out_;
    
    std::ofstream likelihood_out_;
  public:
    map<vertex_type,int> degreeupdatecount;

    /**
     * Create a residual splash engine. Old constructor with residual bound as
     * convergence criterion
     */
    residual_splash_bp(factor_graph_type* factor_graph,
                       size_t splash_size,
                       double bound,
                       double damping) :
      factor_graph_(factor_graph),
      splash_size_(splash_size),
      damping_(damping),
      csr_(sum_product),
      updates_out_("updates.txt"),
      likelihood_out_("likeli.txt") {

      convergence_indicator_ = new residual_splash_convergence_measure(bound);
      own_convergence_indicator_ = true;

      std::cout << "Running Residual Splash BP" << std::endl;
      // Ensure that the factor graph is not null
      assert(factor_graph_ != NULL);
    };

    /**
     * Create a residual splash engine. New constructor with flexible convergence
     * criteria via convergence_indicator pointer.
     */
    residual_splash_bp(factor_graph_type* factor_graph,
                       size_t splash_size,
                       residual_splash_convergence_measure* convergence_indicator,
                       double damping) :
      factor_graph_(factor_graph),
      splash_size_(splash_size),
      damping_(damping),
      csr_(sum_product),
      convergence_indicator_(convergence_indicator),
      own_convergence_indicator_(false),
      updates_out_("updates.txt"){
      std::cout << "Running Residual Splash BP" << std::endl;
      // Ensure that the factor graph is not null
      assert(factor_graph_ != NULL);
    } // end residual_splash_bp

    ~residual_splash_bp(){
      if(own_convergence_indicator_)
        delete convergence_indicator_;
    }


    /**
     * Executes the actual engine.
     */
    void run(int *spcount = NULL, int *upcount = NULL) {
      // initialize the messages
      initialize_state();
      // Ran splash to convergence
      
      int t = splash_to_convergence();
      if (spcount) (*spcount) = t;
      // Close the file stream
      if (upcount) (*upcount) = update_count;
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
      schedule_.clear();
      // double initial_residual = bound_ + (bound_*bound_);
      double initial_residual = std::numeric_limits<double>::max();
      foreach(const vertex_type& u, factor_graph_->vertices()) {
        schedule_.push(u, initial_residual);
      }
    } // end of initialize

    size_t update_count;
    /**
     * Splash to convergence
     */
    int splash_to_convergence() {
      timer ti;
      assert(schedule_.size() > 0);
      size_t splash_count= 0;
      update_count = 0;
      ti.start();
      double outtime = ti.current_time();
      
      convergence_indicator_->start();
      while(!convergence_indicator_->is_converged(schedule_.top().second, 1) ){
        splash(schedule_.top().first);
        if(ti.current_time() - outtime > 2) {
          std::cout << splash_count << ": " << update_count<<": " << schedule_.top().second << std::endl;
          
          finite_assignment f;
          get_map_assignment(f);
          likelihood_out_ << update_count<<", "<<factor_graph_->log_likelihood(f) << std::endl;
          outtime = ti.current_time();
        }
//        update_count++; //this seems to be a redundant increment -anton
      }
      return splash_count;
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
        update_count++;
        send_messages(v);
      }
      // Push belief from the root to the leaves (skipping the
      // root which was processed in the previous pass)
      foreach(const vertex_type& v,
              std::make_pair(++splash_order.begin(), splash_order.end())) {
        update_count++;
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



    /**
     * This writes the new message into the place of the old message
     * and updates the scheduling queue and does any damping necessary
     */
    inline void update_message(const vertex_type& source,
                               const vertex_type& target,
                               const message_type& new_msg) {
      typedef typename factor_type::result_type result_type;

      // Create a norm object (this should be lightweight)
      norm_type norm;
      // Get the original message
      message_type& original_msg = messages_[source][target];
      // Compute the norm
      double new_residual = norm(new_msg, original_msg);
      // Update the residual
      if(schedule_.get(target) < new_residual )
        schedule_.update(target, new_residual);
      // Require that there be no zeros
      assert(new_msg.minimum() > result_type(0.0));
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
      degreeupdatecount[source]++;
      updates_out_ << factor_graph_->vertex2id(source) << std::endl;
      foreach(const vertex_type& dest,
              factor_graph_->neighbors(source)) {
        send_message(source, dest);
      }
      // Mark the vertex as having been visited
      schedule_.update(source, 0.0);
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
        new_msg = new_msg.collapse(make_domain(&target.variable()), csr_.cross_op);
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
    belief_type belief(variable_type* variable) {
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

/**
     * Compute the belief for a vertex
     */
    void belief(std::map<vertex_type, belief_type> &beliefs) {
      // Initialize the belief as uniform
      // For each of the neighbors of the vertex
      foreach(const vertex_type& u,
              factor_graph_->vertices()) {
        belief_type blf;
        if(u.is_factor()) {
          blf = u.factor();
        } else {
          blf = belief_type(make_domain(&(u.variable())),1.0).normalize();
        }
        foreach(const vertex_type& other,
              factor_graph_->neighbors(u)) {
          // Combine the in_msg with the destination factor
          blf.combine_in( messages_[other][u], csr_.dot_op);
        }
        blf.normalize();
        beliefs[u] = blf;
      }
    } // end of send_message
    
    void get_map_assignment(finite_assignment &mapassg) {
      std::map<vertex_type, belief_type> beliefs;
      belief(beliefs);
      foreach(const vertex_type &v, factor_graph_->vertices()) {
        if (v.is_variable()) {
          finite_assignment localmapassg = arg_max(beliefs[v]);
          mapassg[&(v.variable())] = localmapassg[&(v.variable())];
        }
      }
    }
  }; // End of class residual_splash_bp


}; // end of namespace
#include <sill/macros_undef.hpp>



#endif

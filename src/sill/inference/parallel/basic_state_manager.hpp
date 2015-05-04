#ifndef BASIC_STATE_MANAGER_HPP
#define BASIC_STATE_MANAGER_HPP

// STL includes
#include <map>

// SILL Includes
#include <sill/model/factor_graph_model.hpp>
#include <sill/parallel/pthread_tools.hpp>
#include <sill/parallel/timer.hpp>
#include <sill/parallel/binned_scheduling_queue.hpp>
#include <sill/factor/util/norms.hpp>
#include <sill/parallel/object_allocator.hpp>
#include <sill/inference/parallel/message_data.hpp>

// This include should always be last
#include <sill/macros_def.hpp>

/**
 * \file basic_state_manager.hpp
 *
 * Defines the basic message manager which manages the state of a
 * (multicore) BP algorithm
 */
namespace sill {


  /**
   * \class basic_state_manager
   *
   * Manages the state of a belief propagation algorithm Manages the
   * messages and residuals.
   */
  template <typename F>
  class basic_state_manager {
  public:
    typedef F factor_type;

    typedef factor_type message_type;
    typedef factor_type belief_type;
  
    typedef typename factor_type::result_type    result_type;
    typedef typename factor_type::variable_type  variable_type;
    typedef typename factor_type::domain_type    domain_type;

    typedef factor_graph_model<factor_type> factor_graph_model_type;
    typedef typename factor_graph_model_type::vertex_type vertex_type;

  private:

    typedef std::map<vertex_type, 
                     std::map<vertex_type, 
                              MessageData<factor_type> > > message_map_type;
    
    typedef std::map<const variable_type, 
                     std::pair<factor_type,spinlock> > belief_map_type;
    

    /**
     * The underlying factor graph that this algorithm is solving
     */
    const factor_graph_model_type* model_;
    
    /// The vertices that have degree > 0
    std::vector<vertex_type> valid_vertices_;
    
    //! Epsilon of error tollerated for convergence
    double epsilon_;
    bool finished_;

    /**
     * The norm used to evaluate the change in messages.  Here we use
     * an L1 norm to measure the change in factors. 
     */
    factor_norm_1<message_type> norm_;

    object_allocator_tls<message_type> message_buffers_;

    binned_scheduling_queue<vertex_type> schedule_;
  
    /** 
     * Stores the variable to factor and factor to variable messages
     * \note: It may actually be possible to combine these two maps
     * into \n one by making it a map<void*, map<void*, factor_type> >
     */
    message_map_type messages_;
    belief_map_type beliefs_;
    bool allow_simultaneous_rw_;
    size_t update_count_;
  public:  
    //! Preallocate all messages for the graphical model
    basic_state_manager(const factor_graph_model_type* model,
                        double epsilon = 1.0E-5,
                        int num_schedule_queues = 20,
                        bool allow_simultaneous_rw = false) : 
      model_(model), epsilon_(epsilon), finished_(false), 
      schedule_(num_schedule_queues),
      allow_simultaneous_rw_(allow_simultaneous_rw) {

      update_count_ = 0;
      // Ensure that the model point argument is not NULL
      assert(model_ != NULL);
      
      
      std::cerr << model->arguments().size() << " variables" << std::endl;
      timer ti;
      ti.start();
      // iterate over the factors to construct the var to factor
      // and factor to var messages    
      foreach(const factor_type& curfactor, model_->factors()) {
        // iterate over the edges of the factor
        if ( curfactor.arguments().size() ) {
          foreach(variable_type v, curfactor.arguments()) {
            // Message from variable to factor
            messages_[vertex_type(v)][vertex_type(&curfactor)].message = 
              message_type(make_domain(v), 1.0).normalize();
            // message from factor to variable
            messages_[vertex_type(&curfactor)][vertex_type(v)].message = 
              message_type(make_domain(v), 1.0).normalize();
          }
        }
      }
      
      // instantiate the variable beliefs
      foreach(variable_type v, model_->arguments()) {
        //         domain_type tempdomain;
        //         tempdomain.insert(v);
        //         beliefs_[v].first = belief_type(tempdomain,1).normalize();
        beliefs_[v].first = 1;
      }
      
      
      // All vertices must have at least one neighbor (their unary factor)
      //      // count the total number of vertices (for printing purposes only)
      //      size_t totalvertexcount = 0;
      //      // collect all the valid vertices.  
      //       foreach(vertex_type v, model_->vertices()) {
      //         if (model_->num_neighbors(v) > 0) {
      //           valid_vertices_.push_back(v);
      //         }
      //         ++totalvertexcount;
      //       }
      //       std::cerr << valid_vertices_.size() << " out of "
      //                 << totalvertexcount << " vertices with degree > 0"
      //                 << std::endl;


      // Comm
      // double initial_priority = epsilon + (epsilon*epsilon);
      double initial_priority = std::numeric_limits<double>::max();
      schedule_.init(boost::begin(model_->vertices()),  // start iterator
                     boost::end(model_->vertices()),    // end iterator
                     initial_priority);                 // priority
      std::cerr << "Message Initialization took: " << ti.current_time() 
                << std::endl;
    } // end of constructor


  
    /**
     * Returns all the factors associated with a particular variable
     * except unary factors associated with that variable.
     */
    forward_range<const vertex_type&> neighbors(const vertex_type& v) const {
      return model_->neighbors(v);
    }

    /**
     * Returns all the factors associated with a particular variable
     * except unary factors associated with that variable.
     */
    size_t num_neighbors(const vertex_type& v) const {
      return model_->num_neighbors(v);
    }

    forward_range<const vertex_type&> vertices() const {
      return model_->vertices();
    }


    /**
     * This method checks out the messages from the vertex v1 to the
     * vertex v2.  Once a message is checked out it cannot be read or
     * modified by any other thread.  It therefore must eventually be
     * checked in to the manager.  Returns NULL if the message is not
     * found
     */
    message_type* checkout(const vertex_type& v1,
                           const vertex_type& v2,
                           const ReadWrite& rw){
      // find f in the map
      typename message_map_type::iterator i = messages_.find(v1);
      if (i==messages_.end()) return NULL;
    
      // find v in the map
      typename message_map_type::mapped_type::iterator j = i->second.find(v2);
      if (j==i->second.end()) return NULL;
    
      return j->second.checkout(message_buffers_,rw);
    }

  
    /**
     * This method checks out the messages from the vertex v1 to the
     * vertex v2.  Once a message is checked out it cannot be read or
     * modified by any other thread.  It therefore must eventually be
     * checked in to the manager.  Returns NULL if the message is not
     * found.
     * This will return NULL if someone else is holding the write lock
     */
    message_type* try_checkout(const vertex_type& v1,
                               const vertex_type& v2,
                               const ReadWrite& rw){
      // find f in the map
      typename message_map_type::iterator i = messages_.find(v1);
      if (i==messages_.end()) return NULL;
    
      // find v in the map
      typename message_map_type::mapped_type::iterator j = i->second.find(v2);
      if (j==i->second.end()) return NULL;
    
      return j->second.trycheckout(message_buffers_,rw);
    }

  
    
    /**
     * This function checks out the belief of a variable v.  Once a
     * belief is checked out, the caller has exclusive access to it
     * and may not be checked out by any other thread.  Returns NULL
     * of the variable is not found.
     */
    belief_type* checkout_belief(const vertex_type& v){
      assert(v.is_variable());
      // search for the belief
      typename belief_map_type::iterator i = 
        beliefs_.find(&v.variable());
      // Assert that the belief must be present
      assert(i != beliefs_.end());
      // Grab the lock on the belief
      i->second.second.lock();
      // Return a pointer to the belief
      return &(i->second.first);
    }

    /**
     * Use this method to access the belief from the final state
     * manager.  We take a vertex as an argument because eventually,
     * we will store factor beliefs as well.
     */
    const belief_type& belief(const vertex_type& v) const {
      // Get the belief
      assert(v.is_variable()); 
      // search for the belief
      typename belief_map_type::const_iterator i = 
        beliefs_.find(&v.variable());
      // Assert that the belief must be present
      assert(i != beliefs_.end());
      // Return a pointer to the belief
      return (i->second.first);
    }
    
    /**
     * This method checks in the messages from the vertex v1 to the
     * vertex v2. \see checkout
     */
    void checkin(const vertex_type& v1, 
                 const vertex_type& v2,
                 const message_type* msg) {
      MessageData<message_type> *md = &(messages_[v1][v2]);
      double residual = md->checkin(message_buffers_, msg, norm_, 
                                    allow_simultaneous_rw_);
      if (residual > 0) {
        schedule_.promote(v2,residual);
      }
    }

    /**
     * This function checks in the belief of a variable v, allong
     * other threads to check it out.
     */
    void checkin_belief(const vertex_type& v, belief_type* b){
      assert(v.is_variable());
      // search for the belief
      typename belief_map_type::iterator i = 
        beliefs_.find(&v.variable());
      // Assert that the belief must be present
      assert(i != beliefs_.end());
      // Release the lock on the belief
      i->second.second.unlock();
    }
    
  
    /**
     * Gets the top factor and deactivates it from the scheduling
     * queue.  This vertex will not be accessible by deschedule_top
     * until schedule(v) is invoked on that vertex.
     */
    std::pair<vertex_type, double> deschedule_top() {
      return schedule_.deschedule_top();
    }

    /**
     * Call this method to enable v to be run again by deschedule_top
     */
    void schedule(vertex_type v) {
      schedule_.schedule(v);
    }

    /**
     * Call this method to mark a vertex as visited setting its residual
     * to zero.
     */
    void mark_visited(vertex_type v) {
      schedule_.mark_visited(v);
    }

    
  
    //! Gets the factor/variable residual. (Depends on what 'vertex' is).
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

//     //! Gets the termination bound criteria
//     inline double get_bound(){
//       return epsilon_;
//     }

    /**
     * Tests whether the particular vertex is available. This is
     * really only an issue in a distributed implementation where a
     * distant vertex may not be available.
     */
    inline bool available(const vertex_type& v) { return true; }

    /**
     * Determines whether the state of the execution is finished. This
     * will return true if the highest residual message has a residual
     * less than epsilon initialized at construction
     */
    bool finished() {
      if(!finished_) {
        double d = schedule_.top_priority();
        finished_ = d < epsilon_;
        // Display the update progress
                if((update_count_ % 1000) == 0 || finished_) {
                  std::cout << "Progress: " << d << std::endl;
//                   size_t numv_unprocessed = 0;
//                   foreach(const vertex_type& v, valid_vertices_) {
//                     if (residual(v) > 2) {
//                       ++numv_unprocessed;
//                     }
//                   }
//                  std::cout << "Unprocessed V: " << numv_unprocessed << std::endl;          
                }
        update_count_++;
      }
      return finished_;
    }
  }; // End of basic message manager


} // End of namespace

#include <sill/macros_undef.hpp>

#endif

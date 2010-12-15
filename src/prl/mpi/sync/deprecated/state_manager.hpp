#ifndef STATE_MANAGER_HPP
#define STATE_MANAGER_HPP

// STL includes
#include <map>

// PRL Includes
#include <prl/model/factor_graph_model.hpp>
#include <prl/parallel/pthread_tools.hpp>
#include <prl/parallel/timer.hpp>
#include <prl/parallel/binned_scheduling_queue.hpp>
#include <prl/factor/norms.hpp>
#include <prl/parallel/object_allocator.hpp>
#include <prl/inference/parallel/message_data.hpp>

// This include should always be last
#include <prl/macros_def.hpp>

/**
 * \file state_manager.hpp
 *
 * Defines the basic message manager which manages the state of a
 * (multicore) BP algorithm
 */
namespace prl {


  /**
   * \class state_manager
   *
   * Manages the state of a belief propagation algorithm Manages the
   * messages and residuals.
   */
  template <typename F>
  class state_manager {
  public:
    typedef F factor_type;

    typedef factor_type message_type;
    typedef factor_type belief_type;
  
    typedef typename factor_type::result_type    result_type;
    typedef typename factor_type::variable_type  variable_type;
    typedef typename factor_type::domain_type    domain_type;
    // ASSERT that collapse_type = factor_type;
    // TODO we should concept assert this
    typedef typename factor_type::collapse_type  collapse_type;

    typedef factor_graph_model<factor_type> factor_graph_model_type;
    typedef typename factor_graph_model_type::vertex_type vertex_type;
    typedef typename factor_graph_model_type::vertex_id_type vertex_id_type;

    typedef mutable_queue<vertex_type, double> schedule_type;

  private:

    //! IMPORTANT map is in [dest][src] form
    typedef std::map<vertex_type, 
                     std::map<vertex_type, 
                              MessageData<factor_type> > > message_map_type;
    
    typedef std::map<vertex_type, 
                     std::pair<factor_type,spinlock> > belief_map_type;
    

    /**
     * The underlying factor graph that this algorithm is solving
     */
    const factor_graph_model_type* factor_graph_;
    
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

    schedule_type schedule_;
  
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
    state_manager(double epsilon = 1.0E-5,
                  bool allow_simultaneous_rw = false) : 
      epsilon_(epsilon), finished_(false), 
      allow_simultaneous_rw_(allow_simultaneous_rw) {
      update_count_ = 0;
    }
    
    void set_factor_graph(const factor_graph_model_type* model) {
      factor_graph_ = model;
      // Ensure that the model point argument is not NULL
      assert(factor_graph_ != NULL);   
    }

    void initialize(std::set<vertex_id_type>& active) {
      // Remove unnecessary messages
      typedef typename message_map_type::iterator msg_iterator;
      for(msg_iterator iter = messages_.begin(); 
          iter < messages_.end(); ++iter) {
        vertex_id_type dest_id = factor_graph_->vertex2id(iter->first);
        if(active.find(dest_id) == active.end()) messages_.erase(iter);
      }

      // Remove unnecessary beliefs
      typedef typename belief_map_type::iterator blf_iterator;
      for(blf_iterator iter = beliefs_.begin();
          iter < beliefs_.end(); ++iter) {
        vertex_id_type dest_id = 
          factor_graph_->vertex2id(iter.first);
        if(active.find(dest_id) == active.end()) beliefs_.erase(iter);
      }

      // Remove unnecessary residuals
      schedule_.clear();

      // Initialize any message that may be needed
      foreach(vertex_id_type dest_id, active) {
        vertex_type& dest = factor_graph_->id2vertex(dest_id);
        // for all the neighbors of the destination
        foreach(vertex_type& src, factor_graph_->neighbors(dest)) {
          // Get the src2msgdata map
          typedef typename message_map_type::mapped_type mapped_type;
          mapped_type& msgs = messages_[dest];
          // if the source is not defined in this map
          if(msgs.find(src) == msgs.end()) {
            // create the message
            domain_type domain = make_domain(dest.is_variable()?
                                             &(dest.variable()) : 
                                             &(src.variable()) );
            msgs[src].message = message_type(domain, 1.0).normalize();
          }
        }
      } // end of foreach

      // Initialize any beliefs that may be needed
      foreach(vertex_id_type dest_id, active) {
        vertex_type& dest = factor_graph_->id2vertex(dest_id);
        if(dest.is_variable()) {
          beliefs_[dest].first = 1;
        }
        // Initialize the scheduling queue
        schedule_.push(dest, std::numeric_limits<double>::max());
      }
    } // end of initialize


  
    /**
     * Returns all the factors associated with a particular variable
     * except unary factors associated with that variable.
     */
    forward_range<const vertex_type&> neighbors(const vertex_type& v) const {
      return factor_graph_->neighbors(v);
    }


    /**
     * This method checks out the messages from the vertex v1 to the
     * vertex v2.  Once a message is checked out it cannot be read or
     * modified by any other thread.  It therefore must eventually be
     * checked in to the manager.  Returns NULL if the message is not
     * found
     */
    message_type* checkout(const vertex_type& src,
                           const vertex_type& dest,
                           const ReadWrite& rw){
      // find dest in the map
      typedef typename message_map_type::iterator iterator_dest;
      iterator_dest i = messages_.find(dest);
      assert(i !=messages_.end());
    
      // find src in the map
      typedef typename message_map_type::mapped_type::iterator iterator_src;
      iterator_src j = i->second.find(src);
      assert(j != i->second.end());

      message_type* msg = j->second.checkout(message_buffers_,rw);
      assert(msg != NULL);
    
      return msg;
    }

  
    /**
     * This method checks out the messages from the vertex v1 to the
     * vertex v2.  Once a message is checked out it cannot be read or
     * modified by any other thread.  It therefore must eventually be
     * checked in to the manager.  Returns NULL if the message is not
     * found.
     * This will return NULL if someone else is holding the write lock
     */
    message_type* try_checkout(const vertex_type& src,
                               const vertex_type& dest,
                               const ReadWrite& rw){
      // find dest in the map
      typedef typename message_map_type::iterator iterator_dest;
      iterator_dest i = messages_.find(dest);
      assert(i !=messages_.end());
    
      // find src in the map
      typedef typename message_map_type::mapped_type::iterator iterator_src;
      iterator_src j = i->second.find(src);
      assert(j != i->second.end());
    
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
      typename belief_map_type::iterator i = beliefs_.find(v);
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
      typename belief_map_type::const_iterator i = beliefs_.find(v);
      // Assert that the belief must be present
      assert(i != beliefs_.end());
      // Return a pointer to the belief
      return (i->second.first);
    }
    

    /**
     * Use this method to access the belief from the final state
     * manager.  We take a vertex as an argument because eventually,
     * we will store factor beliefs as well.
     */
    belief_type& belief(const vertex_type& v)  {
      // Get the belief
      assert(v.is_variable()); 
      return beliefs_[v].first;
    }
    

    /**
     * This method checks in the messages from the vertex v1 to the
     * vertex v2. \see checkout
     */
    void checkin(const vertex_type& src, 
                 const vertex_type& dest,
                 const message_type* msg) {
      MessageData<message_type>& md = messages_[dest][src];
      double residual = md.checkin(message_buffers_, msg, norm_, 
                                   allow_simultaneous_rw_);
      if (residual > 0 && residual > schedule_.get(dest)) {
        schedule_.update(dest, residual);
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
        beliefs_.find(v);
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
      std::pair<vertex_type, double>& top = schedule_.top();
      // Demote to zero priority
      schedule_.update(top, 0.0);
      return top;
    }

    /**
     * Call this method to enable v to be run again by deschedule_top
     */
    void schedule(vertex_type v) {
      // NOP
    }

    /**
     * Call this method to mark a vertex as visited setting its residual
     * to zero.
     */
    void mark_visited(vertex_type v) {
      schedule_.update(v, 0.0);
    }

    //! Gets the factor/variable residual. (Depends on what 'vertex' is).
    double residual(const vertex_type& v) {
      return schedule_.get(v);
    }
  
    //! Gets the variable residual  
    double residual(const variable_type* v) {
      return residual(vertex_type(v));
    }
  
    //! Gets the factor residual
    double residual(const factor_type* f) {
      return residual(vertex_type(f));
    }

    /**
     * Tests whether the particular vertex is available. This is
     * really only an issue in a distributed implementation where a
     * distant vertex may not be available.
     */
    inline bool available(const vertex_type& v) { 
      return schedule_.contains(v);
    }

    /**
     * Determines whether the state of the execution is finished. This
     * will return true if the highest residual message has a residual
     * less than epsilon initialized at construction
     */
    bool finished() {
      return schedule_.top().second < epsilon_;
    } // end of finished
  }; // End of basic message manager


} // End of namespace

#include <prl/macros_undef.hpp>

#endif

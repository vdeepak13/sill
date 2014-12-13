#ifndef MESSAGE_MANAGER_HPP
#define MESSAGE_MANAGER_HPP

// STL includes
#include <map>

// SILL Includes
#include <sill/model/factor_graph_model.hpp>
#include <sill/parallel/pthread_tools.hpp>
#include <sill/parallel/binned_scheduling_queue.hpp>
#include <sill/factor/util/norms.hpp>


// This include should always be last
#include <sill/macros_def.hpp>

/**
 * \file message_manager.hpp
 *
 * Defines the basic message manager which manages the state of a
 * (multicore) BP algorithm
 */
namespace sill {

  /**
   *  \enum ReadWrite
   *
   * Used by MessageData and message managers to determine if the
   * checkout is meant for reading or writing
   */
  enum ReadWrite { Reading, Writing };
  
  /**
   * \struct object_allocator 
   * 
   * Implements a small pool of objects which users can checkin and
   * checkout.  Checkedin objects are then reused.  This is useful for
   * situations where you need an object for temporary storage, but do
   * not desire to pay the penalty for a malloc.
   *
   * This pool is synchronized and is thread-safe
   */
  template<typename T> 
  class object_allocator{
  private:
    spinlock listlock;
    std::list<T*> freelist;
    std::list<T*> usedlist;
  public:
    /**
     * checks out an object from the pool. An object will not be
     * simultaneously given to more than one user.  If the free list
     * is empty, the default constructor for the object is used to
     * create new objects.
     */
    T* checkout() {
      T* ret;
      listlock.lock();
      if (freelist.size() > 0) {
        ret = freelist.front();
        freelist.pop_front();
      }
      else {
        ret = new T;
      }
      usedlist.push_back(ret);
      listlock.unlock();
      return ret;
    }

    /**
     * Returns an object from the pool. This object can then by given
     * to other users 
     *
     * BUGBUG: This is a linear time operations.  I don't think we
     * want that
     */
    void checkin(T* f) {
      listlock.lock();
      // Linear time operations
      usedlist.remove(f);
      freelist.push_front(f);
      listlock.unlock();
    }
  }; // End of object_allocator
  
  
  /**
   * \class MessageData
   *
   * Handles the synchronizing (checkin and checkout) of a message
   * 
   * This class implements a kind of reader/writer semantics, but take
   * the assumption that there can only be single writer at any one
   * time.  (assertion failure if two writers try to grab it)
   *  
   * checkouts for reads will give it the original message, but
   * checkouts for writes will provide a writebuffer (which will have
   * the same contents).  When the writebuffer is checked in, it will
   * be "committed" by updating the message with the writebuffer.
   *
   * A "pool" object has to be provided to allocate the write buffers
   */
  template<typename F>
  class MessageData {
  public:
    F message;
    spinlock lock;  // Locks the write buffer and the readercount
    F *writebuffer;
    char readercount;
  
    MessageData() {
      writebuffer = NULL;
      readercount = 0;
    }

    /**
     * Checks the message.  If its a write request, a buffer will be
     * taken from the object_allocator
     */
    F *checkout(object_allocator<F> &pool, 
                const ReadWrite rw) {
      if (rw == Reading) {
        lock.lock();
        readercount++;
        lock.unlock();
        return &message;
      }
      else {
        lock.lock();
        // we only support 1 writer. 
        assert(writebuffer == NULL);
        writebuffer = pool.checkout();
        lock.unlock();
        (*writebuffer) = message;

        return writebuffer;
      }
    }


    /**
     * Checks in the message.  If the message was checked out for
     * writing, the buffer will be committed and returned to the
     * object_allocator if allow_simultaneous_rw is not set, the
     * buffer will be committing regardless of whether there is an
     * existing reader.  If it is set, the thread will wait until
     * there are no readers before committing the buffer.  \warning
     * Note that the implementation allows for writer starvation.
     * Returns the norm of the change of value. (using the norm()
     * function) if its a write checkin. Returns 0 otherwise.
     */
    double checkin(object_allocator<F> &pool, 
                   const F *msg, const factor_norm<F>& norm, 
                   const bool allow_simultaneous_rw) {
      double residual = 0;
      lock.lock();
      // if it matches message address, this was a read request
      if (msg == &message) {
        assert(readercount > 0);
        readercount--;
      }
      //this was a write request
      else if (msg == writebuffer) {
        //Optional: Wait for readers to complete
        if (!allow_simultaneous_rw) {
          while(readercount > 0) {
            lock.unlock();
            sched_yield();
            lock.lock();
          }
        }
        // compute the delta
        residual = norm(message , *(writebuffer));
        message = *(writebuffer);
        pool.checkin(writebuffer);
        writebuffer = NULL;
      }
      else {
        //Erroneous message!
        assert(0);
      }
      lock.unlock();
      return residual;
    }
  }; // End of message_data



  /**
   * \class basic_message_manager
   *
   * Manages the state of a belief propagation algorithm Manages the
   * messages and residuals.
   */
  template <typename F>
  class basic_message_manager {
  public:
    typedef F factor_type;

    typedef factor_type message_type;
    typedef factor_type belief_type;
  
    typedef typename factor_type::result_type    result_type;
    typedef typename factor_type::variable_type  variable_type;
    typedef typename factor_type::domain_type    domain_type;
  
  private:

    typedef std::map<const variable_type*, 
                     std::map<const factor_type*, 
                              MessageData<factor_type> > > var_to_factor_type;
    
    typedef std::map<const factor_type*, 
                     std::map<const variable_type*, 
                              MessageData<factor_type> > > factor_to_var_type;

    typedef std::map<const variable_type*, 
                     std::pair<factor_type,spinlock> > belief_map_type;
            
    //! The norm used to evaluate the change in messages
    factor_norm_inf<message_type> norm_;

    object_allocator<message_type> message_buffers_;

    binned_scheduling_queue<const variable_type*> var_schedule_;
    binned_scheduling_queue<const factor_type*> factor_schedule_;
  
    /** 
     * Stores the variable to factor and factor to variable messages
     *   \note: It may actually be possible to combine these two maps
     *   into \n one by making it a map<void*, map<void*, factor_type>
     *   >
     */
    var_to_factor_type var_to_factor_;
    factor_to_var_type factor_to_var_;
    belief_map_type beliefs_;
    bool allow_simultaneous_rw_;

  public:  
    //! Preallocate all messages for the graphical model
    basic_message_manager(const sill::factor_graph_model<factor_type> &model,
                          bool allow_simultaneous_rw = true,
                          int num_schedule_queues = 20) : 
      var_schedule_(1), factor_schedule_(1),
      allow_simultaneous_rw_(allow_simultaneous_rw) {

      // iterate over factors (no node factors)
      std::vector<std::pair<const factor_type* ,double> > schedule_f;
    
      foreach(const factor_type& curfactor, model.factors()) {
        schedule_f.push_back(std::make_pair(&curfactor,100));
        // iterate over the edges of the factor
        if (curfactor.arguments().size() > 1) {
          foreach(variable_type* v, curfactor.arguments()) {
            /*
             * TODO: little issue here. I will actually need a way to
             *  initialize messages. The current method only works
             *  table factors.
             */
            //var to factor message has the same size as the factor
            var_to_factor_[v][&curfactor].message = 
              factor_type(curfactor.arguments(), 1).normalize();

            domain_type tempdomain;
            tempdomain.insert(v);
            factor_to_var_[&curfactor][v].message = 
              factor_type(tempdomain,1).normalize();
          }
        }
      }
      // divide by 2 because I have 2 schedule queues
      factor_schedule_.init(num_schedule_queues/2, schedule_f);
    
      // instantiate the beliefs
      std::vector<std::pair<const variable_type* ,double> > schedule_v;
      foreach(variable_type* v, model.arguments()) {
        domain_type tempdomain;
        tempdomain.insert(v);
        beliefs_[v].first = belief_type(tempdomain,1).normalize();
        schedule_v.push_back(std::make_pair(v,100));
      }
      var_schedule_.init(num_schedule_queues/2, schedule_v); 
    }
  
    /**
     * This method checks out the messages from the variable v to the
     * factor f.  Once a message is checked out it cannot be read or
     * modified by any other thread.  It therefore must eventually be
     * checked in to the manager.  Returns NULL if the message is not
     * found
     */
    factor_type* checkout_variable_to_factor(const variable_type* v,
                                             const factor_type* f, 
                                             const ReadWrite rw){
      // find v in the map
      typename var_to_factor_type::iterator i = var_to_factor_.find(v);
      if (i==var_to_factor_.end()) return NULL;
    
      // find f in the map
      typename var_to_factor_type::mapped_type::iterator j = i->second.find(f);
      if (j==i->second.end()) return NULL;
    
      return j->second.checkout(message_buffers_,rw);
    }


    /**
     * This method checks out the messages from the factor f to the
     * variable v.  Once a message is checked out it cannot be read or
     * modified by any other thread.  It therefore must eventually be
     * checked in to the manager.  Returns NULL if the message is not
     * found
     */
    message_type* checkout_factor_to_variable(const factor_type* f,
                                              const variable_type* v,
                                              const ReadWrite rw){
      // find f in the map
      typename factor_to_var_type::iterator i = factor_to_var_.find(f);
      if (i==factor_to_var_.end()) return NULL;
    
      // find v in the map
      typename factor_to_var_type::mapped_type::iterator j = i->second.find(v);
      if (j==i->second.end()) return NULL;
    
      return j->second.checkout(message_buffers_,rw);
    }

  
    /**
     * This function checks out the belief of a variable v.  Once a
     * belief is checked out, the caller has exclusive access to it
     * and may not be checked out by any other thread.  Returns NULL
     * of the variable is not found.
     */
    message_type* checkout_belief(const variable_type* v){
      // search for the belief
      typename belief_map_type::iterator i = beliefs_.find(v);
      if (i==beliefs_.end()) return NULL;
    
      i->second.second.lock();
    
      return &(i->second.first);
    }

    /**
     * This method checks in the messages from the variable v to the
     * factor f.  Once a message is checked out it cannot be read or
     * modified by any other thread.  It therefore must eventually be
     * checked in to the manager.
     */
    void checkin_variable_to_factor(const variable_type* v, 
                                    const factor_type* f,
                                    const message_type* msg) {
      MessageData<message_type> *md = &(var_to_factor_[v][f]);
      double residual = md->checkin(message_buffers_, msg, norm_, 
                                    allow_simultaneous_rw_);
      if (residual > 0) {
        factor_schedule_.promote(f,residual);
      }
    }

    /**
     * This method checks in the messages from the factor f to the
     * variable v.  Once a message is checked out it cannot be read or
     * modified by any other thread.  It therefore must eventually be
     * checked in to the manager.
     */
    void checkin_factor_to_variable(const factor_type* f,
                                    const variable_type* v,
                                    const message_type* msg) {
      MessageData<message_type> *md = &(factor_to_var_[f][v]);
      double residual = md->checkin(message_buffers_, msg, norm_, 
                                    allow_simultaneous_rw_);
      if (residual > 0) {
        var_schedule_.promote(v,residual);
      }
    }
  
    /**
     * This function checks in the belief of a variable v, allong
     * other threads to check it out.
     */
    void checkin_belief(const variable_type* v, const belief_type* b){
      beliefs_[v].second.unlock();
    }

    /** 
     * Activates a factor. Its residual will go to 0 and will not be
     * scheduled again until it is deactivated. Its residual can still
     * be increased, but it will have no effect until it is
     * deactivated.
     */
    void activate(const factor_type* f) {
      factor_schedule_.deschedule(f);
    }

    /**
     * Deactivates a factor.  \see activate
     */  
    void deactivate(const factor_type* f) {
      factor_schedule_.schedule(f);
    }
  
    /** 
     * Activates a variable. Its residual will go to 0 and will not be
     * scheduled again until it is deactivated. Its residual can still
     * be increased, but it will have no effect until it is
     * deactivated.
     */
    void activate(const variable_type* v) {
      var_schedule_.deschedule(v);
    }
  
    /** 
     * Deactivates a variable factor.  \see deactivate
     */  
    void deactivate(const variable_type* v) {
      var_schedule_.schedule(v);
    }
  
    /**
     * Gets the top factor and activates it
     */
    std::pair<const factor_type*,double> get_top_factor() {
      return factor_schedule_.deschedule_top();
    }
  
    /**
     * Gets the top variable and activates it
     */
    std::pair<const variable_type*,double> get_top_variable() {
      return var_schedule_.deschedule_top();
    }
  
    //! Gets the variable residual  
    double residual(const variable_type* v) {
      return var_schedule_[v];
    }
  
    //! Gets the factor residual
    double residual(const factor_type* v) {
      return var_schedule_[v];
    }

  }; // End of basic message manager

} // End of namespace


#include <sill/macros_undef.hpp>

#endif

#ifndef MPI_STATE_MANAGER_HPP
#define MPI_STATE_MANAGER_HPP

// forward declaration
namespace sill {
namespace mpi_state_manager_prot {
template <typename F>
class mpi_state_manager_protocol;
}
}

#include <sys/wait.h>
#include <unistd.h>
// STL includes
#include <map>

// SILL Includes
#include <sill/model/factor_graph_model.hpp>
#include <sill/parallel/pthread_tools.hpp>
#include <sill/parallel/timer.hpp>
#include <sill/parallel/binned_scheduling_queue.hpp>
#include <sill/factor/util/norms.hpp>
#include <sill/inference/parallel/message_data.hpp>
#include <sill/inference/parallel/mpi_state_manager_protocol.hpp>
#include <sill/model/mooij_kappen_derivatives.hpp>
#include <sill/model/factor_graph_partitioning.hpp>

// This include should always be last
#include <sill/macros_def.hpp>

/**
 * \file mpi_state_manager.hpp
 *
 * Defines the basic message manager which manages the state of a
 * cluster BP algorithm
 */
 
namespace sill {
  using namespace mpi_state_manager_prot;
  
  
   /**
      Design specs:
      
      (Also see mpi_state_manager_protocol.hpp. There is a design spec section)
      
      The mpi_state_manager is broken into 2 parts:
        1: mpi_state_manager
        2: mpi_state_manager_protocol
        
      They work closely together and in fact the mpi_state_manager_protocol
      is a friend of the mpi_state_manager and frequently goes in and touch
      deep datastructures.
      
      Essentially, mpi_state_manager_protocol is an interface to the 
      mpi_state_manager that runs through MPI. If the mpi_state_manager wants
      to send a message, it just calls the appropriate send_message function
      in the mpi_state_manager_protocol. You will find that very few arguments 
      are needed for the send_message function because the 
      mpi_state_manager_protocol actually digs in the mpi_state_manager to 
      find the information it has to send.
      
      Similarly, if a message is received, mpi_state_manager_protocol gets the
      message, and does the appropriate digging in mpi_state_manager to complete
      the task.
      
      Very roughly, the internal layout of mpi_state_manager is very similar
      to the basic_state_manager. 
      There is : 
        - A collection of messages, each wrapped in a MessageData structure 
          which handles the individual synchronization of a message
        - A collection of belief vectors (which are rarely touched)
        - A pointer to a factor_graph_model which contains the complete
          structure of the network
        - A priority queue to store residuals and handle scheduling.

      The key difference is that the local node only has a subset of the 
      entire set of possible messages. Therefore message updates to messages
      I do not own, has to be temporarily buffered, and sent out over the 
      network when checked in. 


      Nodes are allowed to give each other vertices. An issue is that this
      requires exotic locking if it is to be 100% safe. 
      
      Synchronization is therefore a big problem. There are 3 possible sources 
      of contention
        1: some number of splash threads
        2: receive thread in mpi_state_manager_protocol
        3: Other hypothetical threads (like maybe a thread which negotiates
           vertex exchanges?)
      (See the TODO comment in mpi_state_manager_protocol::give_away_vertex() 
      for ideas for solving this)
   */
  
  /**
   * \class mpi_state_manager
   *
   * Manages the state of a belief propagation algorithm Manages the
   * messages and residuals.
   *
   * The MPI_state_manager initialization is broken into 4 stages
   * with 'acknowledges' seperating the stages
   *
   * Stage 1: block slicing and graph distribution
   *          - The master node is provided the graph as a factor_graph_model
   *          - Graph is sliced into blocks by the master node
   *          - Master creates finite_variable* -> int and vice versa
   *            mapping table
   *          - BP and graph parameters, and mapping table are transmitted 
   *            to all nodes
   * -----------------------ACK---------------------------
   * Stage 2:  - vertex->block mapping and graph are serialized and transmitted
   *            to all nodes 
   *           - block allocation and message creation
   *          - Each node creates the messages and residuals for the blocks 
   *            it own
   * -----------------------ACK---------------------------
   * Stage 3: BP
   *          - BP starts.
   *          - When a node has max residual < bound, it sends a "stopped" 
   *            signal to the master node and stops computation. 
   *            If its max residual (due to the effect of other
   *            incoming messages) increases beyond the bound, it sends a 
   *            "resumed" signal to the master node and resumes computation.
   *          - When master node determines that all nodes have max residual <
   *            bound, it waits for 5 seconds to ensure that there are no other
   *            messages in flight. Then it issues a "compute belief" signal
   *          - All nodes compute beliefs on each vertex
   * -----------------------ACK---------------------------
   * Stage 4: collect beliefs
   *          - All nodes send the beliefs to the master node
   *          - master node writes out the beliefs
   * -----------------------BARRIER---------------------------
   *      All programs terminate
   * ---------------------------------------------------------
   *
   *
   * Construction is broken into 2 parts. Calling the constructor will perform
   * the first half of stage 1 (ending at slicing). Construction will also
   * register the MPI handler.
   *
   * The caller must then call mpi_post_office::start(), then call the second
   * half of the construction mpi_state_manager::start().
   *
   * This is necessary because I cannot register handlers after starting the 
   * post office.
   *
   * NOTE: TODO:
   * vertex<->id mappings are in factor_graph_model
   * variable<-> id mappings are in the state manager
   * Why?
   * It just turned out that way because factor_graph_model already has a 
   * vector of vertices and I didn't want to replicate that locally.
   */
  template <typename F>
  class mpi_state_manager {
  public:
    friend class mpi_state_manager_protocol<F>;
    typedef F factor_type;

    typedef factor_type message_type;
    typedef factor_type belief_type;
    
    typedef typename factor_type::result_type    result_type;
    typedef typename factor_type::variable_type  variable_type;
    typedef typename factor_type::domain_type    domain_type;

    typedef factor_graph_model<factor_type> factor_graph_model_type;
    typedef typename factor_graph_model_type::vertex_type vertex_type;
    typedef std::pair<vertex_type, vertex_type> edge_type;
    
    struct deferred_insertion_data{
      vertex_type v;
      belief_type belief; //only valid if v is a variable
      double priority;
    };
  private:
    typedef MessageDataUnsynchronized<factor_type> message_data_type;
    typedef std::map<vertex_type, 
                     std::map<vertex_type, 
                     message_data_type> > message_map_type;
    
    typedef std::map<const variable_type, belief_type> belief_map_type;
    


    /**
     * The underlying factor graph that this algorithm is solving
     */
    factor_graph_model_type* model_;

    //! Epsilon of error tollerated for convergence
    double epsilon_;

    //execution status
    size_t update_count_;
    
    // stores the local state of finished_
    bool finished_;
    
    /**
     * The norm used to evaluate the change in messages.  Here we use
     * an L1 norm to measure the change in factors. 
     */
    factor_norm_1<message_type> norm_;    
    multi_object_allocator_tls<message_type> message_buffers_;
  
    /**
     stores all the messages 
     the lock must be used with great care
     
     deleting stuff from the message map is unsafe (even while locked)
     since someone else may want it later. 
     
     inserting stuff into the message is safe if you lock it
     */
    mutex state_lock_;
    message_map_type messages_;
    belief_map_type beliefs_;
    
    /// defered message transmissions
    std::map<edge_type, message_type> deferred_transmissions;
    /// deferred schedule insertions
    mutex deferred_insertions_lock;
    std::list<deferred_insertion_data> deferred_insertions;
    std::map<edge_type, message_type> deferred_receives;
    
    //! the maximum number of blocks I am allowed to own
    size_t max_vertices_;
    
    //! converts a vertex_type to the owner of the vertex
    std::map<vertex_type, uint32_t> vertex2owner_;
    //! gets a list of vertices for each owner
    std::vector<std::set<vertex_type> > owner2vertex_;
    
    // maps variables to a globally unique id and vice versa
    std::map<variable_type, uint32_t> var2id_;
    std::vector<variable_type> id2var_;    
    
    mpi_post_office &po_;
    // protocol handler 
    // NOTE: We do not free this! mpi_post_office owns the pointer!
    mpi_state_manager_protocol<F> *mpiprot_;
    bool ismaster_;
    
    // master uses this finished counter to keep track of how many are done
    // a lock is necessary as 2 threads may write to it. (myself and the MPI 
    // receive thread). Though the operation should be atomic anyway
    size_t finishedcounter_; 
    mutex finishedcounter_mut_;
    conditional finishedcounter_cond_;

    
    // finish() checks this flag if shutdown is true it will return true
    // otherwise finish() will halt on the condition variable
    bool shutdown_;
    mutex shutdown_mut_;
    conditional shutdown_cond_;
    
    // TODO: use a real priority queue
    binned_scheduling_queue<vertex_type> schedule_;

    
    
 
    /**
      constructs all messages terminating at a vertex I own
    */
    void ConstructLocalMessages() {
      // iterate over the factors to construct the var to factor
      // and factor to var messages    
      int nummessagescreated = 0; 
      foreach(const vertex_type &v, owner2vertex_[po_.id()]) {
        // iterate over the edges of the factor
        if(model_->num_neighbors(v) > 0) {
          foreach(const vertex_type &w, model_->neighbors(v)) {
          
            domain_type tempdomain;
            if (v.is_variable()) tempdomain.insert(&(v.variable()));
            else if (w.is_variable()) tempdomain.insert(&(w.variable()));

            //var to factor message and factor to var messages 
            // are the same size as the var 
            
            messages_[w][v].message = factor_type(tempdomain, 1).normalize();
            nummessagescreated++;
          }
        }
        if (v.is_variable()) {
          domain_type vdomain;
          vdomain.insert(&(v.variable()));
          beliefs_[&(v.variable())] = belief_type(vdomain,1).normalize();
        }

      }
      std::cout << "Messages created " << nummessagescreated << std::endl;
    }
    
    /// should only be called by the root
    void ConstructAllBeliefs() {
      foreach (variable_type *v, model_->arguments()) {
        domain_type vdomain;
        vdomain.insert(v);
        beliefs_[v] = belief_type(vdomain,1).normalize();
      }
    }
    
    void ConstructVarIdMapping() {
      var2id_.clear();
      id2var_.clear();
      id2var_.resize(model_->arguments().size());
      int id = 0;
      foreach (variable_type *v, model_->arguments()) {
        var2id_[v] = id;
        id2var_[id] = v; 
        id++;
      }
    }
    
    void CreateSchedule() {
        schedule_.init(boost::begin(owner2vertex_[po_.id()]),  // start iterator
                    boost::end(owner2vertex_[po_.id()]),    // end iterator
                    100);
    }
    
    void SendBeliefs() {
      foreach(const vertex_type &v, owner2vertex_[po_.id()]) {
        if (v.is_variable()) {
          mpiprot_->send_belief(0, v);
        }
      }
    }
    
    void SetOwner(const vertex_type &v, const size_t newowner) {
      int oldowner = vertex2owner_[v];
      vertex2owner_[v] = newowner;
      owner2vertex_[oldowner].erase(v);
      owner2vertex_[newowner].insert(v);
    }
  public:  
    /**
      Constructor. 
      mpi_post_office must be provided.
      
      If this is node 0, model must not be NULL.
      If this is not node 0, all other parameters are ignored.
    */
    mpi_state_manager(mpi_post_office &po,
                      factor_graph_model_type* model = NULL,
                      double epsilon = 1.0E-5, 
                      size_t max_vertices_per_node = 100) : 
      model_(model), epsilon_(epsilon), update_count_(0), finished_(false),
      max_vertices_(max_vertices_per_node), po_(po),
      schedule_(50) {  // TODO: use a real priority queue
      
      // registers with MPI
      mpiprot_ = new mpi_state_manager_protocol<F>(*this, po_);
      po_.register_handler(MPI_STATE_MANAGER_PROT_ID,
                           mpiprot_);
      std::cout << "handler registered" << std::endl;

      ismaster_ = (po_.id() == 0);
      // model!=NULL if and only if I am the master node
      assert(ismaster_ == (model_!=NULL));

      if (ismaster_) {
        if (access("pmetis", F_OK) == 0) {
          SliceGraphMetis(*model, po_.num_processes(), 
                          owner2vertex_, vertex2owner_, true);
        }
        else {
          std::cout << "pmetis not found!, Defaulting to BFS cut.\n";
          SliceGraphBFS(*model, po_.num_processes(), owner2vertex_, vertex2owner_);
        }
        ConstructVarIdMapping();
      }
    }
    
    ~mpi_state_manager() {
      // cleanup
      /*for(typename belief_map_type::iterator i = beliefs_.begin();
          i != beliefs_.end();
          ++i) {
        delete i->second;
      }
      
      for(typename message_map_type::iterator i = messages_.begin();
          i != messages_.end();
          ++i) {
          for(typename message_map_type::mapped_type::iterator j = i->second.begin();
              j != i->second.end();
              ++j) {
            delete j->second;
          }
      }*/
      delete mpiprot_;
    }
    void start() {
      if (ismaster_) {
        // stage 1
        mpiprot_->send_message(MTYPE_PARAMETERS);
        // stage 2
        // other nodes will call their respective Construct Local Messages
        // and Create Schedule
        mpiprot_->send_message(MTYPE_FACTOR_GRAPH);
        ConstructLocalMessages();
        CreateSchedule();
        ConstructAllBeliefs();
      }
      finishedcounter_ = 0;
      std::cout << "Init Complete Barrier" << std::endl;
      MPI::COMM_WORLD.Barrier();
    }
  
    /**
     * Returns all the factors associated with a particular variable
     * except unary factors associated with that variable.
     */
    forward_range<const vertex_type&> neighbors(const vertex_type& v) const {
      return model_->neighbors(v);
    }

    size_t num_neighbors(const vertex_type& v) const {
      return model_->num_neighbors(v);
    }


    forward_range<const vertex_type&> vertices() const {
      return model_->vertices();
    }

    /**
     * Returns all the unary factors associated with a particular
     * variable. If none is associated this will return a NULL pointer
     */
    const factor_type* node_factor(variable_type v) const {
      return model_->node_factor(v);
    }



    /**
     * This method checks out the messages from the vertex v1 to the
     * vertex v2.  Once a message is checked out it cannot be read or
     * modified by any other thread.  It therefore must eventually be
     * checked in to the manager.  Returns NULL if the message is not
     * found
     *
     * If this is a message that will end up going to another machine
     * the only possibility is to checkout for writing. assertion 
     * failure will result otherwise
     */
    message_type* checkout(const vertex_type& v1,
                           const vertex_type& v2,
                           const ReadWrite& rw){
        state_lock_.lock();  
      // NOTE: This has to be changed if we ever run multiple threads per node
      // NOTE: This has to be changed if we allow for asynchronous removal of vertices
      
      // NOTE: its for a remote vertex
      // since there can be only one write....
      // just use the message buffer since we know that 
      // it definitely will not be used
      if (vertex2owner_[v2] != po_.id()){
        assert(rw == Writing);
        if(rw == Writing) {
          // its for remote. Therefore it MUST be for writing
          message_type* msg = message_buffers_.checkout();
          if (v1.is_variable()) (*msg) = message_type(make_domain(&(v1.variable())),1);
          else if (v2.is_variable()) (*msg) = message_type(make_domain(&(v2.variable())),1);
          
          state_lock_.unlock();
          return msg;
        }
        else return NULL; // to eliminate a warning 
      } else {
        // its for reading a
        /*if (vertex2owner_[v1] == po_.id()) {
          std::cout << "boundary co\n";
        }*/
        typename message_map_type::iterator i = messages_.find(v1);
        if (i==messages_.end()) {
          state_lock_.unlock();
          return NULL; 
        }
        // find v2 in the map
        typename message_map_type::mapped_type::iterator j = i->second.find(v2);
        if (j==i->second.end()) {
          state_lock_.unlock();
          return NULL;   
        }
        message_type* mt =  j->second.checkout(message_buffers_,rw);
        state_lock_.unlock();
        return mt;
      }
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
      // NOTE: This has to be changed if we ever run multiple threads per node
      return checkout(v1,v2,rw);
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
      // Return a pointer to the belief
      return &(i->second);
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
      return (i->second);
    }
    
    
    /**
     * This method checks in the messages from the vertex v1 to the
     * vertex v2. \see checkout
     */
    void checkin(const vertex_type& v1, 
                 const vertex_type& v2,
                 const message_type* msg) {
      state_lock_.lock();
      //check if its for remote
      if (vertex2owner_[v2] != po_.id()) {
        // it is!
        // do a remote write out
        deferred_transmissions[edge_type(v1,v2)] = *msg;
        // mpiprot_->send_message_update(v1, v2, *msg);
        message_buffers_.checkin(msg);
      }
      else {
      /*  if (vertex2owner_[v1] == po_.id()) {
          std::cout << "boundary ci\n";
        }*/
        message_data_type *md = &(messages_[v1][v2]);
        double residual = md->checkin(message_buffers_, msg, norm_, false);
        if (residual >= 0) {
          schedule_.promote(v2,residual);
        }
      }
      state_lock_.unlock();
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
      // if I don't own the message, return -1
      if (vertex2owner_[v] != po_.id()) return -1;
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
     * Tests whether the particular vertex is available. This is
     * really only an issue in a distributed implementation where a
     * distant vertex may not be available.
     */
    inline bool available(const vertex_type& v) { 
      return vertex2owner_[v] == po_.id();
    }

    bool SlaveFinishHandler() {
      static int calls = 0;
      /* this loop is complicated.
          it works like this
          1: Check if I am done.locally. If my state changes update the root
          2: if I am not done locally, resume execution
          3: if I am done locally:
            3a: if shutdown tells me to shutdown, we are done
            3b: otherwise, we stop execution and block until someone
                wakes us up. 
                if someone wakes us up it can either mean
                  - we received a new message
                  - someone set the shutdown flag
                therefore we go back into the loop and check everything again
      */
      while(1) {
        // this is my local belief that i am done
        bool prevfinished = finished_;
        double d = schedule_.top_priority();
        if (calls % 5000 == 0) {
          std::cout << d << std::endl;
        }
        calls++;
        finished_ = d < epsilon_;
        if (prevfinished == false && finished_ == true) {
          //std::cout << "finished\n";
          mpiprot_->send_message(MTYPE_FINISHED, 0);
        }
        else if (prevfinished == true && finished_ == false) {
          //std::cout << "resumed\n";
          mpiprot_->send_message(MTYPE_RESUMED, 0);
        }
        if (finished_ == false) return false;
        else {
          shutdown_mut_.lock();
          if (shutdown_) {
            shutdown_mut_.unlock();
            mpiprot_->PrintCounts();
            return true;
          }
          shutdown_cond_.wait(shutdown_mut_);
          shutdown_mut_.unlock();
        }
      }
    }

    bool MasterFinishHandler() {
      static int calls = 0;
        // I am the root
      // this is similar but we don't check the shutdown flag
      // instead we check the finished counter. if finished counter == # procs - 1
      // we are done
      int donecount = 0;
      while(1) {
        // this is my local belief that i am done
        double d = schedule_.top_priority();
        if (calls % 5000 == 0) {
          std::cout << d << std::endl;
        }
        calls++;
  
        finished_ = d < epsilon_;
        if (finished_ == false) return false;
        else {
          // std::cout << "finished\n";
          finishedcounter_mut_.lock();
          if(finishedcounter_ == po_.num_processes() - 1) { // done
            donecount++;
            std::cout << "done " << donecount << "\n";
            if (donecount == 2) {
              mpiprot_->send_message(MTYPE_SHUTDOWN, -1);
              finishedcounter_mut_.unlock();
              mpiprot_->PrintCounts();
              return true;
            }
          }
          else { // not done
            donecount = 0;
          }
        }
        finishedcounter_mut_.unlock();
  
        // this will sleep for 1 seconds, or if someone sends me
        // a BP message or a FINISH or a RESUME
        finishedcounter_cond_.timedwait(finishedcounter_mut_, 1);
        finishedcounter_mut_.unlock();
      }
    }

    /**
     * Determines whether the state of the execution is finished. This
     * will return true if the highest residual message has a residual
     * less than epsilon initialized at construction
     */
    bool finished() {   
      // handle deferred sends
      typedef std::pair<edge_type, message_type> deferredtype;
      foreach(deferredtype e, deferred_transmissions) {
        mpiprot_->send_message_update(e.first.first, e.first.second, e.second);
      }
      deferred_transmissions.clear();
            
      // handle the second stage of vertex insertion
      deferred_insertions_lock.lock();
      typedef std::pair<vertex_type, double> vertexresidualpair;
      foreach(deferred_insertion_data &e, deferred_insertions) {
        schedule_.push(e.v, e.priority);
        domain_type vdomain;
        // std::cout << "deferred insertion of vertex." << std::endl;
        if (e.v.is_variable()) {
          beliefs_[&(e.v.variable())] = e.belief;
          assert(e.belief.arg_vector().size() > 0);
        }
        SetOwner(e.v, po_.id());
        // its ok to make this uniform beliefs
      }
      deferred_insertions.clear();
      state_lock_.lock();
      foreach(deferredtype e, deferred_receives) {
        messages_[e.first.first][e.first.second].message = e.second;
      }
      state_lock_.unlock();
      deferred_receives.clear();
      deferred_insertions_lock.unlock();

      // handle deferred receives
      typedef std::pair<edge_type, message_type> deferredtype;
      foreach(deferredtype e, deferred_transmissions) {
        mpiprot_->send_message_update(e.first.first, e.first.second, e.second);
      }
      deferred_transmissions.clear();
      // handle termination condition
      if (po_.id() != 0) {
        return SlaveFinishHandler();
      }
      else {
        return MasterFinishHandler();
      }
    }
    
    
    void collect_beliefs() {
      mpiprot_->send_message(MTYPE_SEND_BELIEF,-1);
    }
    
    size_t num_vertices_owned() {
      return owner2vertex_[po_.id()].size();
    }
    
    bool owner(vertex_type v) {
      return (vertex2owner_[v] == po_.id());
    }
    
    void print() {
      std::cout << "Epsilon: " << epsilon_ << "\n";
      std::cout << "Max vertices: " << max_vertices_ << "\n";
      std::cout << "Num vertices: " << num_vertices_owned() << "\n";
      if (model_ == NULL) {
        std::cout << "Graph: NULL\n";
      }
      else {
        model_->print(std::cout);
      }
    }
    void print_variable_mapping() {
      for (size_t i = 0;i < id2var_.size(); ++i) {
        std::cout << i << ": " << id2var_[i] << std::endl;
      }
    }
  }; // End of basic message manager

} // End of namespace


#include <sill/macros_undef.hpp>

#endif // MPI_STATE_MANAGER_HPP

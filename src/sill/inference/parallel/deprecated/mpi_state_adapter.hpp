#ifndef MPI_STATE_ADAPTER_HPP
#define MPI_STATE_ADAPTER_HPP

// STL includes
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <sstream>
#include <limits>


// Boost libraries
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>


// PRL Includes
#include <sill/serializable.hpp>
#include <sill/mpi/mpi_wrapper.hpp>
#include <sill/mpi/mpi_consensus.hpp>
#include <sill/parallel/pthread_tools.hpp>
#include <sill/model/factor_graph_model.hpp>
#include <sill/parallel/timer.hpp>
#include <sill/factor/util/norms.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/datastructure/mutable_queue.hpp>
#include <sill/inference/parallel/message_data.hpp>


// This include should always be last
#include <sill/macros_def.hpp>




#include <sill/macros_def.hpp>

namespace sill {

  class mpi_state_adapter {
    ///////////////////////////////////////////////////////////////////////
    // typedefs
  public:
    typedef tablef factor_type;
    typedef factor_type message_type;
    typedef factor_type belief_type;
    typedef factor_type::domain_type domain_type;

    typedef factor_graph_model<factor_type> factor_graph_type;
    typedef factor_graph_type::variable_type variable_type;
    typedef factor_graph_type::vertex_type vertex_type;
    typedef factor_graph_type::vertex_id_type vertex_id_type;

    typedef uint32_t owner_id_type;
    typedef uint32_t variable_id_type;
    typedef std::set<vertex_type> vertex_set_type;

    typedef std::pair<vertex_type, vertex_type> directed_edge;


    typedef factor_norm_1<message_type> norm_type;

    ///////////////////////////////////////////////////////////////////////
    // private typedefs
  private:

    typedef std::pair<message_type, mutex> message_lock_pair;

    //! type of internal message maps
    typedef std::map<directed_edge, message_lock_pair > message_map_type;

    //! type of belief map
    typedef std::map<vertex_type, belief_type> belief_map_type;
    
    //! the residual priority queue type
    typedef mutable_queue<vertex_type, double> schedule_type;

    //! a map from the local variable pointer to its unique id (number)
    typedef std::map<variable_type*, variable_id_type> var2id_map_type;

    //! a map from the variables unique id to its local pointer
    typedef std::vector<variable_type*> id2var_map_type;

    // Callbacks are all friends
    friend class bp_message_callback;

    /**
     * Callback for bp messages 
     */
    class bp_message_callback : public mpi_post_office::po_box_callback {
      mpi_state_adapter* adapter_;
    public:
      bp_message_callback() : adapter_(NULL) {  }

      void set_adapter(mpi_state_adapter* adapter) { 
        adapter_ = adapter;
      }

      /**
       * This funciton is resposible for receiving bp messages
       * 1) deserialization the message
       * 2) place it in the state manager
       */
      void recv_message(const mpi_post_office::message& mpi_msg) {
        assert(adapter_ != NULL);
        assert(adapter_->factor_graph_ != NULL);

        boost::iostreams::stream<boost::iostreams::array_source> 
          strm(mpi_msg.body, mpi_msg.body_size);
        boost::archive::text_iarchive arc(strm);
        // Deserialize the ids first
        vertex_id_type src_id, dest_id;
        arc >> src_id;  
        arc >> dest_id;
        // Get the actual vertex for source and dest (in the local
        // representation)
        vertex_type src, dest;
        src = adapter_->factor_graph_->id2vertex(src_id);
        dest = adapter_->factor_graph_->id2vertex(dest_id);
        // Get the message from the adapter
        message_type* bp_msg = 
          adapter_->checkout(src, dest, Writing);
        // save the actual message
        bp_msg->load_remap(arc, adapter_->id2var_ );
        // Checkin the message (which will update the priority queue)
        adapter_->checkin(src, dest, bp_msg);
      }
      void terminate() {
        //NOP
      }
    }; // end of bp_message_callback

    


    ///////////////////////////////////////////////////////////////////////
    // Data members
  private:    
    //! pointer to the post office used for mpi communication
    mpi_post_office* po_;

    //! pointer to the factor graph 
    factor_graph_type* factor_graph_;

    //! converts a vertex_type to the owner of the vertex
    std::map<vertex_type, owner_id_type> vertex2owner_;

    //! gets a list of vertices for each owner
    std::map<owner_id_type, vertex_set_type> owner2vertex_set_;

    //! a map from the local variable pointer to its unique id (number)
    var2id_map_type var2id_;

    //! a map from the variables unique id to its local pointer
    id2var_map_type id2var_;

    //! a callback to receive bp messages
    bp_message_callback bp_callback_;
  
    //! the messages
    message_map_type messages_;

    //! the beliefs
    belief_map_type beliefs_;
    
    //! the scheduling queue
    schedule_type schedule_;
    //! mutex required to touch the scheduling queue
    mutex schedule_mutex_;

    //! the termination state
    bool finished_;

    //! global mutex
    mutex mut_;

    //! used to establish consensus
    consensus* consensus_;

  public:

    // NOT COPYABLE OR ASSINGABLE SHOULD MAKE PRIVATE
    mpi_state_adapter(mpi_post_office* po, consensus* consensus) : 
      po_(po), factor_graph_(NULL), consensus_(consensus) { 
      assert(po_ != NULL);
      assert(consensus_ != NULL);
      // set the bp callback to point to this object
      bp_callback_.set_adapter(this);
      // set the bp callback handler
      po_->register_handler(MPI_PROTOCOL::BP_MESSAGE, &bp_callback_);
      // initialize the consensus mechanism
    }

    /**
     * destructor to unregister the handler
     */
    ~mpi_state_adapter() {
      // set the bp callback handler
      po_->unregister_handler(MPI_PROTOCOL::BP_MESSAGE);
    }    


    /**
     * This is invoked on the root node at startup. The argument
     * provides the source factor graph on which inference is to be
     * run.
     */
    void init(factor_graph_type& factor_graph) {
      size_t root_node = 0;
      assert(po_->id() == root_node);
      // We will need a local copy here
      factor_graph_ = &factor_graph;
      // Slice the graph
      slice_factor_graph();
      // construct variable mappings for transmission
      construct_varid_mapping();
      // initialize the scheduling queue
      init_schedule();
      init_messages();
      // broadcast factor graph and parameters
      std::stringstream strm;
      boost::archive::text_oarchive arc(strm);
      // Send the factor graph
      arc << *factor_graph_;
      // Send the mappings
      arc << vertex2owner_;
      arc << owner2vertex_set_;
      arc << var2id_;
      arc << id2var_;
      // Broadcast the message to all other nodes using the mpi bcast
      size_t body_size = strm.str().length();
      char* body = const_cast<char*>(strm.str().c_str());
      // This will block until the graph is transmitted
      // Transmit the size of the datastructure
      po_->mpi_bcast(body_size, root_node);
      // Transmit the graph
      po_->mpi_bcast_message(body_size,
                             body, 
                             root_node);
      // wait for everyone to finish init
      po_->barrier();
    } // end of init called from root

    /**
     * This is invoked on all other nodes except the root node at
     * startup.
     */
    void init() {
      size_t root_node = 0;
      assert(po_->id() != root_node);
      // here we will need to simply get the graph from another node
      size_t body_size = 0;
      // This should block until the graph is received
      po_->mpi_bcast(body_size, root_node);
      // allocate enough space to store the body
      char* body = new char[body_size];
      po_->mpi_bcast_message(body_size, body, root_node);   
   
      // deserialize the factor graph and maps
      boost::iostreams::stream<boost::iostreams::array_source> 
        strm(body, body_size);
      boost::archive::text_iarchive arc(strm);
      // Resurect the factor graph
      factor_graph_ = new factor_graph_type();
      arc >> *factor_graph_;
      // Resurect the mappings
      arc >> vertex2owner_;
      arc >> owner2vertex_set_;
      arc >> var2id_;
      arc >> id2var_;
      // Delete the body data
      delete [] body;
      // Initialize the schedule
      init_schedule();
      init_messages();
      // wait for everyone to finish init
      po_->barrier();
    } // end of init called from not root


    void collect_beliefs() {
      assert(po_ != NULL);
      assert(factor_graph_ != NULL);
      size_t root_node = 0;

      // Serialize all the beliefs
      std::stringstream strm;
      boost::archive::text_oarchive arc(strm);
      foreach(vertex_type v, owner2vertex_set_[po_->id()]) {
        if(v.is_variable()) {
          // get the vertex id
          vertex_id_type v_id = factor_graph_->vertex2id(v);
          // get the belief
          const belief_type& blf = belief(v);
          // do serialization
          arc << v_id;
          blf.save_remap(arc, var2id_);
        }
      }

      // Gather the ammount of data stored on each node
      int local_size = strm.str().size();
      int* buffer_sizes = NULL;
      if(po_->id() == root_node) {
        buffer_sizes = new int[po_->num_processes()];
      }
      MPI::COMM_WORLD.Gather(&local_size, 1, MPI::INT,
                             buffer_sizes, po_->num_processes(), MPI::INT,
                             root_node);
      // assert that if this is root buffer_sizes contains the ammount
      // of data each node will send
      
      // Construct the offsets
      int* offsets = NULL;
      char* recv_buffer = NULL;
      size_t recv_buffer_size = 0;
      if(po_->id() == root_node) {
        offsets = new int[po_->num_processes()];
        offsets[0] = 0;
        recv_buffer_size = buffer_sizes[0];
        for(size_t i = 1; i < po_->num_processes(); ++i) {
          offsets[i] = offsets[i-1] + buffer_sizes[i-1];
         recv_buffer_size += buffer_sizes[i];
        }
        recv_buffer = new char[recv_buffer_size];
      }

      // Collect all the factors
      MPI::COMM_WORLD.Gatherv(strm.str().c_str(), local_size, MPI::CHAR,
                              recv_buffer, buffer_sizes, offsets, MPI::CHAR,
                              root_node);

      // If this is root node then resurect all beliefs locally
      if(po_->id() == root_node) {
        // Deserialize and save all the beliefs
        boost::iostreams::stream<boost::iostreams::array_source> 
          strm(recv_buffer, recv_buffer_size);
        boost::archive::text_iarchive arc(strm);
        while(strm.good()) {
          // Deserialize the ids first
          vertex_id_type vertex_id;
          arc >> vertex_id;  
          // Get the local vertex corresponding to the id
          vertex_type vertex = factor_graph_->id2vertex(vertex_id);
          // Get the local belief
          belief_type* blf = checkout_belief(vertex);
          // Update the belief
          blf->load_remap(arc, id2var_);
          // Checkin the modified belief (currently a NOP)
          checkin_belief(vertex, blf);
        } // end of while loop

        std::cout << "Belief Size: " << beliefs_.size() << std::endl;
      } // end of if root resurection


      // Free all memory buffers
      if(buffer_sizes != NULL) delete [] buffer_sizes;
      if(offsets != NULL) delete [] offsets;
      if(recv_buffer != NULL) delete [] recv_buffer;
            
    } // end of collect beliefs


    const factor_graph_type& factor_graph() {
      assert(factor_graph_ != NULL);
      return *factor_graph_;
    }

    ///////////////////////////////////////////////////////////////////////
    // state manager type routines  
    /**
     * Returns all the factors associated with a particular variable
     * except unary factors associated with that variable.
     */
    forward_range<const vertex_type&> 
    neighbors(const vertex_type& v) const {
      assert(factor_graph_ != NULL);
      return factor_graph_->neighbors(v);
    } // end of neighbors


    /**
     * very wimpy checkout
     */
    message_type* checkout(const vertex_type& v1,
                           const vertex_type& v2,
                           const ReadWrite& rw){
      assert(po_ != NULL);
      assert(po_->id() >= 0);
      // Require than we own the vertex
      assert(vertex2owner_[v2] == po_->id());
      // if we are checking out for writing just make a new message
      // and return it
      if(rw == Writing) {
        // construct the domain of this message.  we know that either
        // v1 or v2 represents a variable (and the other represents a
        // factor).
        domain_type domain = 
          make_domain(v2.is_variable()? &(v2.variable()) : &(v1.variable()));
        return new message_type(domain, 1.0);
      } else {
        // Otherwise it for reading only and so we use the currently stored
        // message
        message_lock_pair& pair = messages_[directed_edge(v1,v2)];
        message_type* msg = &(pair.first);
        pair.second.lock();
        return msg;
      }
    } // end of checkout

    /**
     * same as standard checkout except that it returns null of the
     * current message is in use.
     */
    message_type* try_checkout(const vertex_type& v1,
                               const vertex_type& v2,
                               const ReadWrite& rw){
      assert(po_ != NULL);
      // if we don't own the destination vertex then simply return null
      if(vertex2owner_[v2] != po_->id()) return NULL;
      // if the vertex is for writing then we just make a duplicate
      if(rw == Writing) {
        // construct the domain of this message.  we know that either
        // v1 or v2 represents a variable (and the other represents a
        // factor).
        domain_type domain = 
          make_domain(v2.is_variable()? &(v2.variable()) : &(v1.variable()));
        return new message_type(domain, 1.0);
      } else {
        message_lock_pair& pair = messages_[directed_edge(v1,v2)];
        message_type* msg = &(pair.first);
        if(pair.second.try_lock()) return msg;
        else return NULL;
      }
    } // end of try_checkout


    /**
     * This function checks out the belief of a variable v.  Once a
     * belief is checked out, the caller has exclusive access to it
     * and may not be checked out by any other thread.  Returns NULL
     * of the variable is not found.
     */
    belief_type* checkout_belief(const vertex_type& v){
      assert(v.is_variable());
      return &(beliefs_[v]);
    } // end of checkout_belief

    /**
     * Use this method to access the belief from the final state
     * manager.  We take a vertex as an argument because eventually,
     * we will store factor beliefs as well.
     */
    const belief_type& belief(const vertex_type& v) const {
      // Get the belief
      assert(v.is_variable());
      typedef belief_map_type::const_iterator iterator;
      iterator iter = beliefs_.find(v);
      assert(iter != beliefs_.end());
      return iter->second;
    } // end of belief 


    /**
     * This method checks in the messages from the vertex v1 to the
     * vertex v2. \see checkout
     */
    void checkin(const vertex_type& v1, 
                 const vertex_type& v2,
                 message_type* msg) {
      assert(po_ != NULL);
      // If we don't own the denstination vertex then we simply send
      // the message to the approriate owner and free the mememory
      if(vertex2owner_[v2] != po_->id()) {
        transmit_message(v1,v2,msg);
        // delete the message
        delete msg;
      } else {
        // Otherwise we own the message so we must update internal
        // data structures
        message_lock_pair& pair = messages_[directed_edge(v1,v2)];
        message_type& internal_msg(pair.first);
        // determine if the message being returned was the original or
        // a duplicate.  If the message was the original then it was
        // checked out for reading so we must free the lock
        if(&internal_msg == msg)  {
          pair.second.unlock();
        } else {
          norm_type norm;
          // if the message was a duplicate then it was checked out
          // for writing. 
          pair.second.lock();
          double residual = norm(*msg, internal_msg);
          internal_msg = *msg;
          pair.second.unlock();
          // Update the schedule if necessary
          schedule_mutex_.lock();
          if(schedule_.get(v2) < residual) schedule_.update(v2, residual);
          schedule_mutex_.unlock();        
          // delete the message
          delete msg;
        }
      }
    } // end of checkin

    /**
     * This function checks in the belief of a variable v, allong
     * other threads to check it out.
     */
    void checkin_belief(const vertex_type& v, belief_type* b){
      // NOP right now
    }
    
  
    /**
     * Deschedule the top vertex
     */
    std::pair<vertex_type, double> deschedule_top() {
      typedef std::pair<vertex_type, double> pair_type;
      schedule_mutex_.lock();
      const pair_type& pair(schedule_.top());
      schedule_mutex_.unlock();
      return pair;
    }

    /**
     * Call this method to enable v to be run again by deschedule_top
     */
    void schedule(vertex_type& v) {
      // do nothing
    }

    /**
     * Call this method to mark a vertex as visited setting its residual
     * to zero.
     */
    void mark_visited(const vertex_type& v) {
      schedule_mutex_.lock();
      schedule_.update(v, 0.0);
      schedule_mutex_.unlock();
    }

    //! Gets the factor/variable residual. (Depends on what 'vertex' is).
    double residual(const vertex_type& v) {
      if(vertex2owner_[v] != po_->id()) {
        return 0;
      } else {
        schedule_mutex_.lock();
        double residual = schedule_.get(v);
        schedule_mutex_.unlock();
        return residual;
      }
    }


    /**
     * Tests whether the particular vertex is available. This is
     * really only an issue in a distributed implementation where a
     * distant vertex may not be available.
     */
    inline bool available(const vertex_type& v) { 
      return vertex2owner_[v] == po_->id();
    }


    /**
     * returns whether bp has converged
     */
    bool finished() {
      schedule_mutex_.lock();
      double max_residual = schedule_.top().second;
      schedule_mutex_.unlock();
      return consensus_->finished(max_residual);
    }    

    ///////////////////////////////////////////////////////////////////////
    // helper functions    
  private:
    void slice_factor_graph() {
      assert(factor_graph_ != NULL);
      assert(po_ != NULL);
      // create a list of all unassigned variables
      std::set<vertex_type> unassigned;
      
      foreach(const vertex_type &v, factor_graph_->vertices()) {
        unassigned.insert(v);
      }
      size_t maxblocksize = (unassigned.size() / po_->num_processes()) + 1;

      // Populate the owner2vertex and vertex2owern maps by doing a
      // breadth first traversal
      for(owner_id_type owner_id = 0; 
          !unassigned.empty() && owner_id < po_->num_processes(); 
          ++owner_id) {  
        std::list<vertex_type> queue;    // Breadth first queue 
        std::set<vertex_type>  visited;  // Set of visited vertices
        std::set<vertex_type>  block;    // set of vertices in the block

        // While the task is still too small and their remains unassigned
        // vertices
        while(block.size() < maxblocksize && !unassigned.empty()) {
          // If no element in queue then start a new tree
          if(queue.empty()) { 
            queue.push_front(*unassigned.begin());
            visited.insert(*unassigned.begin());
          }
          assert(!queue.empty());
          // Pop the first element off the queue 
          vertex_type v = queue.front(); queue.pop_front();
          // Add the element to the task
          block.insert(v);          
          // Remove the vertex from the set of unassigned vertices
          unassigned.erase(v); 
          // Add all its unassigned and unvisited neighbors to the queue
          foreach(vertex_type u, factor_graph_->neighbors(v)) {
            if(unassigned.find(u) != unassigned.end() &&
              visited.find(u) == visited.end()) {
              queue.push_back(u);
              visited.insert(u);
            }
          } // end of add neighbors for loop
        } // End of block build foor loop
        // update owner2vertex_set 
        owner2vertex_set_[owner_id] = block;
        // ensure that the owner_id is valid
        assert(owner_id < po_->num_processes());
        // update vertex2owner 
        foreach(const vertex_type &v, block) {
          vertex2owner_[v] = owner_id;
        }
      }
    } // end of slice factor graph

    void construct_varid_mapping() {
      assert(factor_graph_ != NULL);
      var2id_.clear();
      id2var_.clear();
      id2var_.resize(factor_graph_->arguments().size());
      // enumerate the variables
      variable_id_type id = 0;
      foreach(variable_type *v, factor_graph_->arguments()) {
        var2id_[v] = id;
        id2var_[id] = v; 
        id++;
      }
    } // end of construct_varid_mapping

    /**
     * Initialize the schedule with infinite residual on each vertex
     * for all vertices owned by this process.
     */
    void init_schedule() {
      assert(po_ != NULL);
      foreach(const vertex_type& v, owner2vertex_set_[po_->id()]) {
        schedule_.push(v, std::numeric_limits<double>::max());
      }
    } // end of init_schedule

    /**
     * Initialize all the messages
     */
    void init_messages() {
      foreach(vertex_type dest, owner2vertex_set_[po_->id()]) {
        foreach(vertex_type src, factor_graph_->neighbors(dest)) {
          domain_type domain = 
            make_domain(src.is_variable()? 
                        &(src.variable()) : &(dest.variable()));
          messages_[directed_edge(src,dest)].first = 
            message_type(domain, 1.0).normalize();
        }
      }
    } // end of initialize messages




    /**
     * Used to transmit a bp message over mpi
     */
    void transmit_message(const vertex_type& v1, 
                          const vertex_type& v2,
                          message_type* msg) {
      assert(po_ != NULL);
      vertex_id_type src_id = factor_graph_->vertex2id(v1);
      vertex_id_type dest_id = factor_graph_->vertex2id(v2);
      owner_id_type owner_id = vertex2owner_[v2];
      // Serializae the bp message
      std::stringstream strm;
      boost::archive::text_oarchive arc(strm);
      assert(factor_graph_ != NULL);
      arc << src_id;
      arc << dest_id;
      msg->save_remap(arc, var2id_);
      // Send the message using the post office
      po_->send_message(owner_id,
                        MPI_PROTOCOL::BP_MESSAGE,
                        strm.str().length(),
                        strm.str().c_str());

    } // end of transmit message

  }; // end of mpi_state_adapter







}; // end of namespace


#include <sill/macros_undef.hpp>



#endif





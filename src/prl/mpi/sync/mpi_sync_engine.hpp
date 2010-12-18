#ifndef MPI_SYNC_ENGINE_HPP
#define MPI_SYNC_ENGINE_HPP

#include <sys/wait.h>
#include <unistd.h>


// STL includes
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <sstream>
#include <limits>


// MPI Libraries
#include <mpi.h>

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
#include <sill/model/factor_graph_model.hpp>
#include <sill/factor/norms.hpp>
#include <sill/math/gdl_enum.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/datastructure/mutable_queue.hpp>


// Should eventually define these elsewhere
#define MPI_ROOT_NODE 0
#define BP_MESSAGE_TYPE 20

// This include should always be last
#include <sill/macros_def.hpp>
namespace sill {

  class mpi_sync_engine {
    ///////////////////////////////////////////////////////////////////////
    // typedefs
  public:
    typedef tablef factor_type;
    typedef factor_type message_type;
    typedef factor_type belief_type;
    typedef factor_type::domain_type domain_type;

    typedef factor_graph_model<factor_type>     factor_graph_type;
    typedef factor_graph_type::variable_type    variable_type;
    typedef factor_graph_type::vertex_type      vertex_type;
    typedef factor_graph_type::vertex_id_type   vertex_id_type;

    typedef uint32_t owner_id_type;
    typedef uint32_t variable_id_type;
    typedef std::set<vertex_type> vertex_set_type;

    typedef std::pair<vertex_type, vertex_type> directed_edge;

    typedef factor_norm_1<message_type> norm_type;
    typedef std::vector<vertex_type> ordering_type;


    //! The schedule of vertices
    typedef mutable_queue<vertex_type, double> schedule_type;

    
    //! IMPORTANT map is in [dest][src] form
    typedef std::map<vertex_type, 
                     std::map<vertex_type, message_type> > 
    message_map_type;

    //! Map from vertex to the belief
    typedef std::map<vertex_type, factor_type> belief_map_type;


    /** simple buffer for storing character arrays in C style */
    struct buffer_type { 
      size_t size; 
      char* body; 
      buffer_type(size_t size = 0, char* body = NULL) : 
        size(size), body(body) {}
      void free() {
        assert(body != NULL); 
        assert(size > 0);
        delete [] body;
        body = NULL;
        size = 0;
      }
      void alloc(size_t new_size) {
        if( body != NULL || size > 0 ) free();
        size = new_size;
        body = new char[size];
      }
    };


    //! a map from the local variable pointer to its unique id (number)
    typedef std::map<variable_type*, variable_id_type> var2id_map_type;

    //! a map from the variables unique id to its local pointer
    typedef std::vector<variable_type*> id2var_map_type;


    /** represents a chunk of a graph */
    struct chunk_type {
      std::set<vertex_id_type> interior;
      std::set<vertex_id_type> fringe;
      
      template<typename Archive>
      void serialize(Archive& arc, const unsigned int version) {
        arc & interior;
        arc & fringe;
      }
    };

    /** the type of map from ownerids to chunks */
    typedef std::vector<chunk_type> owner2chunk_type;

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
    belief_map_type beliefs_;

    //! a map from the local variable pointer to its unique id (number)
    var2id_map_type var2id_;

    //! a map from the variables unique id to its local pointer
    id2var_map_type id2var_;

    //! the set of chunks
    owner2chunk_type chunks_;

    //! the size of a splash
    size_t splash_size_;

    //! the size of the fringe
    size_t fringe_size_;

    //! convergence bound
    double bound_;

    //! level of damping 1.0 is fully damping and 0.0 is no damping
    double damping_;

    //! the commutative semiring for updates (typically sum_product)
    commutative_semiring csr_;

    //! temporary message
    message_type new_msg_;

    //! orderinger
    std::vector<vertex_type> splash_order_;

  public:

    /**
     * Create a map reduce engine.  This should be invoked on all nodes.
     */
    mpi_sync_engine(factor_graph_type* factor_graph,
                    size_t splash_size,
                    size_t fringe_size,
                    double bound,
                    double damping) :
      factor_graph_(factor_graph), 
      splash_size_(splash_size), 
      fringe_size_(fringe_size),
      bound_(bound), 
      damping_(damping),
      csr_(sum_product) {

//       std::cout << "Staring Init!" << std::endl;
//       std::cout << "Splash Size: " << splash_size_ << std::endl
//                 << "Fringe Size: " << fringe_size_ << std::endl
//                 << "Bound:       " << bound_ << std::endl
//                 << "Damping:     " << damping_ << std::endl;

      // broadcast the factor graph
      if(id() == MPI_ROOT_NODE) {
        // Ensure that we have the factor graph if we are the root node
        assert(factor_graph != NULL);
        // Broadcast the factor graph
        send_factor_graph();
      } else {
        // We are not the root node so receive the factor graph
        recv_factor_graph();
      }
//       std::cout << "Variables: " << factor_graph_->arguments().size()
//                 << std::endl;
//       std::cout << "Factors: " << factor_graph_->size()
//                 << std::endl;
//       std::cout << "Finished Init!" << std::endl;
//      MPI::COMM_WORLD.Barrier();
      // repartition the graph
      repartition();     
    } // end mpi_sync_engine


    ///////////////////////////////////////////////////////////////////////
    // Initialization
    /**
     * This is invoked on the root node at startup. The argument
     * provides the source factor graph on which inference is to be
     * run.
     */
    void send_factor_graph() {
      assert(id() == MPI_ROOT_NODE);
      assert(factor_graph_ != NULL);
      // Construct varid mappings
      var2id_.clear();
      id2var_.clear();
      id2var_.resize(factor_graph_->arguments().size());
      size_t id = 0;
      foreach(variable_type *v, factor_graph_->arguments()) {
        var2id_[v] = id;
        id2var_[id] = v; 
        id++;
      } // end of foreach
      // Serialize all mappings and factor graph and send to all other
      // nodes
      std::stringstream strm;
      boost::archive::text_oarchive arc(strm);
      // Send the factor graph
      arc << *factor_graph_;
      // Send the mappings
      arc << var2id_;
      arc << id2var_;
      // Braodcast the factor graph
      mpi_bcast(strm.str().c_str(), strm.str().size());
    } // end of init called from root


    /**
     * This is invoked on all other nodes except the root node at
     * startup.
     */
    void recv_factor_graph() {
      // This will allocate and return a buffer containing the factor
      // graph data
      buffer_type buffer = mpi_bcast();
      assert(buffer.body != NULL);
      assert(buffer.size > 0);
      // deserialize the factor graph and maps
      boost::iostreams::stream<boost::iostreams::array_source> 
        strm(buffer.body, buffer.size);
      boost::archive::text_iarchive arc(strm);
      // Resurect the factor graph
      factor_graph_ = new factor_graph_type();
      arc >> *factor_graph_;
      // Resurect the mappings
      arc >> var2id_;
      arc >> id2var_;
      // Delete the body data
      buffer.free();
    } // end of init called from not root




    ///////////////////////////////////////////////////////////////////////
    // Algorithm
    /**
     * Executes the actual engine.  This should be invoked on all
     * nodes and will not terminate until execution has finished
     */
    void start() {
      // initialize the messages
      reinitialize_state();
      // Run the algorithm
      double original_bound = bound_;
      size_t max_iter = 1;
      for(size_t i = 0; i < max_iter; ++i) {
	bound_ = std::pow(original_bound, (double)(i+1.0)/max_iter);
	if(id() == 0) {
	  std::cout << "Bound: " << bound_ << std::endl;
	}
	splash_to_convergence();
	MPI::COMM_WORLD.Barrier();      
	// Exchange messages
	timer time; 
	time.start();
	exchange_messages(chunks_);
	if(id() == 0) {
	  std::cout << "Sync time: " << time.current_time() << std::endl;
	}
      }
      bound_ = original_bound;

      // compute all the beliefs
      chunk_type& local_chunk = chunks_[id()];
      foreach(const vertex_id_type& v_id, local_chunk.interior) {
        vertex_type v = factor_graph_->id2vertex(v_id);
        if(v.is_variable()) update_belief(v);
      } 

      // Collect the beliefs
      if(id() == MPI_ROOT_NODE) {
        // Receive beliefs
        recv_beliefs();
      } else {
        // Send beliefs to root
        send_beliefs();
      }
      // Force all processes to return simultaneously 
      //    MPI::COMM_WORLD.Barrier();      
    } // End of run
    

    /**
     * This function removes unecessary messages and beliefs and
     * reconstructs the schedule
     */
    void reinitialize_state() {
      chunk_type& local_chunk = chunks_[id()];
      // Create the set of vertices which consists of the local chunk
      // and and fringe
      std::set<vertex_id_type> active = local_chunk.interior;
      active.insert(local_chunk.fringe.begin(), 
                    local_chunk.fringe.end());

//       // Remove unnecessary messages
//       typedef message_map_type::iterator msg_iterator;
//       for(msg_iterator iter = messages_.begin(); 
//           iter != messages_.end(); ++iter) {
//         vertex_id_type dest_id = factor_graph_->vertex2id(iter->first);
//         if(active.find(dest_id) == active.end()) messages_.erase(iter);
//       }

//       // Remove unnecessary beliefs
//       typedef belief_map_type::iterator blf_iterator;
//       for(blf_iterator iter = beliefs_.begin();
//           iter != beliefs_.end(); ++iter) {
//         vertex_id_type v_id = 
//           factor_graph_->vertex2id(iter->first);
//         if(active.find(v_id) == active.end()) beliefs_.erase(iter);
//       }

      // Clear schedule
      schedule_.clear();

      // Initialize the schedule
      // double initial_residual = bound_ + (bound_*bound_);
      double initial_residual = std::numeric_limits<double>::max();
      foreach(vertex_id_type v_id, active) {
        vertex_type v = factor_graph_->id2vertex(v_id);
        schedule_.push(v, initial_residual );
      }
    } // end of initialize

    
    /** 
     * This function synchronizes across all processors and
     * repartitions the entire graph. It then automatically exchanges
     * messages to accomodate the new partitioning.
     */
    void repartition() {
      owner2chunk_type new_chunks(num_processes()); 
      // If this is root node then create and send chunks
      if(id() == MPI_ROOT_NODE) {
        slice_factor_graph(new_chunks);
        // Add the fringe to each of the chunk
        foreach(chunk_type& chunk, new_chunks){
          add_fringe(chunk);
        }
        // scatter the Transmit each chunk
        mpi_send_bcast(new_chunks);
      } else { // Otherwise just receive chunks
        // Recv the new chunks
        mpi_recv_bcast(new_chunks);
      }
      // 2) Exchange messages if we have already run once.  The first
      // time this is invoked there is no need to exchange messages as
      // the current owner2chunk mapping is empty.  However on later
      // iterations we may need to send messages to other mappers.
      // The exchange_messages function will populate the current
      // state manager with the necessary information from other
      // mappers
      if(!chunks_.empty()) 
        exchange_messages(new_chunks);
      chunks_ = new_chunks;
    } // end of repartition


    /**
     * Slice factor graph into chunks.  Note that only the interior is
     * constructed here.
     */
    void slice_factor_graph(owner2chunk_type& chunks) {
      assert(factor_graph_ != NULL);
      // create a set of all unassigned variables
      std::set<vertex_type> unassigned;
      foreach(const vertex_type &v, factor_graph_->vertices()) {
        unassigned.insert(v);
      }
      // Compute the maximum size of a block
      size_t maxblocksize = (unassigned.size() / chunks.size()) + 1;
      // Populate the chunks
      for(size_t owner = 0; 
          !unassigned.empty() && owner < chunks.size(); 
          ++owner) {  
        // Clear the chunk interior
        chunks[owner].interior.clear();
        // Initialize data structures
        std::list<vertex_type> queue;    // Breadth first queue 
        std::set<vertex_type>  visited;  // Set of visited vertices
        // While the task is still too small and their remains unassigned
        // vertices
        while(chunks[owner].interior.size() < maxblocksize 
              && !unassigned.empty()) {
          // If no element in queue then start a new tree
          if(queue.empty()) { 
            queue.push_front(*unassigned.begin());
            visited.insert(*unassigned.begin());
          }
          assert(!queue.empty());
          // Pop the first element off the queue 
          vertex_type v = queue.front(); queue.pop_front();
          vertex_id_type v_id = factor_graph_->vertex2id(v);
          // Add the element to the chunk
          chunks[owner].interior.insert(v_id);          
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
        assert(chunks[owner].interior.size() <= maxblocksize);
        assert(chunks[owner].interior.empty() == false); // SHOULD REMOVE THIS?
      } // end of for loop over owners
      assert(unassigned.empty());
    } // end of slice factor graph


    /** cuts up the factor graph using repeated BFS's and fills in
        the structures vertex2owner_, block2owner_ and total_block_count_
    */
   void slice_factor_graph_metis(owner2chunk_type& chunks) {
      // create a list of all unassigned variables
      size_t edge_count = 0;
      foreach(const vertex_type& v, factor_graph_->vertices()) {
        edge_count += factor_graph_->num_neighbors(v);
      }
      assert(edge_count % 2 == 0);
      edge_count = edge_count / 2;

      // Save the file
      const char* tmp_fname = "metistmp.txt";
      std::ofstream fout(tmp_fname);
      int num_vertices = factor_graph_->arguments().size() + 
	factor_graph_->size();
      fout << num_vertices  << " " << edge_count << std::endl;   
      foreach(const vertex_type& u, factor_graph_->vertices()) {
        foreach(const vertex_type& v, factor_graph_->neighbors(u)) {
	  // Metis uses ids starting from 1
          fout << factor_graph_->vertex2id(v) + 1 << " ";
        }
        fout << std::endl;
      }
      fout.close();

      // Execute metis on the file
      std::stringstream num_chunks_strm;
      num_chunks_strm << chunks.size();
      pid_t pid = fork();
      if (pid == 0) {
        execl("pmetis", "pmetis", "metistmp.txt", 
	      num_chunks_strm.str().c_str(), (char*)(0));
      } else {
        int stat;
        waitpid(pid, &stat, 0);
	assert(stat);
      }

      // now read back the output file
      // output will be in metistmp.txt.numparts
      std::stringstream in_filename;
      in_filename << tmp_fname << ".part." << chunks.size();
      std::cout << "reading " << in_filename.str() << std::endl;
      std::ifstream fin(in_filename.str().c_str());
      if (fin.good() == false) {
        std::cerr << "unable to read partition file!" << std::endl;
        getchar();
      }
      
      foreach(const vertex_type& u, factor_graph_->vertices()) {
        owner_id_type owner_id;
        fin >> owner_id;
	chunks[owner_id].interior.insert(factor_graph_->vertex2id(u));
      }
      fin.close(); 
      std::cout << "Partition complete:" << std::endl;
      for(size_t i = 0; i < chunks.size(); ++i) {
	std::cout << i << ": " << chunks[i].interior.size() 
		  << std::endl;
      }
    }


    /**
     * constructs the fringe for a given chunk
     */
    void add_fringe(chunk_type& chunk) {
      assert(factor_graph_ != NULL);
      // temporarily add the entire interior
      chunk.fringe = chunk.interior;
      for(size_t i = 0; i < fringe_size_; ++i) {
        foreach(const vertex_id_type& u_id, chunk.fringe) {
          vertex_type u = factor_graph_->id2vertex(u_id);
          foreach(const vertex_type& v, factor_graph_->neighbors(u))  {
            vertex_id_type v_id = factor_graph_->vertex2id(v);
            chunk.fringe.insert(v_id);
          }
        }
      }
      // Remove the interior
      foreach(vertex_id_type id, chunk.interior) {
        chunk.fringe.erase(id);
      }
    } // end of add fringe


    /**
     * This function sends and receives messages from and to the local
     * state manager. 
     */
    void exchange_messages(owner2chunk_type& new_chunks) {
      chunk_type& my_current_chunk = chunks_[id()];
      chunk_type& my_future_chunk = new_chunks[id()];
      // here I will use the language of sending a vertex when in fact
      // I really mean sending all the messages inbound to that
      // vertex.

      // Compute the vertices that I need to send and which nodes I
      // need to send them too
      typedef std::map< owner_id_type, std::set<vertex_id_type> > 
        send_map_type;
      send_map_type send_map;
      size_t send_count = 0;
      for(size_t owner_id = 0; owner_id < new_chunks.size(); ++owner_id) { 
        // Don't send to self
        if(owner_id != id()) {
          chunk_type& chunk = new_chunks[owner_id];
          // If I currently have the vertex in my interior that is 
          // in the interior of this other chunk then I need to 
          // send that vertex to that other owner
          foreach(vertex_id_type v_id, chunk.interior) {
            if(my_current_chunk.interior.count(v_id) > 0) {
              send_map[owner_id].insert(v_id);
              send_count++;
            }
          }
          // If I currently have the vertex in my interior that is in
          // the fringe of another chunk then I must send that vertex
          // to the other chunk
          foreach(vertex_id_type v_id, chunk.fringe) {
            if(my_current_chunk.interior.count(v_id) > 0) {
              send_map[owner_id].insert(v_id);
              send_count++;
            }
          }
        }
      } // end for loop 
//       std::cout << "Sending: " << send_count << " vertices." << std::endl;

      // Compute the owners from which I must receive vertices
      std::set<owner_id_type> recv_set;
      for(size_t owner_id = 0; owner_id < chunks_.size(); ++owner_id) { 
        // Dont receive from self
        if(owner_id != id()) {
          chunk_type& chunk = chunks_[owner_id];
          // if any of the vertices in the interior of the other chunk
          // are either in the interior or fring of my chunk then I
          // will need to receive them
          foreach(vertex_id_type v_id, chunk.interior) {
            if(my_future_chunk.interior.count(v_id) > 0 ||
               my_future_chunk.fringe.count(v_id) > 0) {
              recv_set.insert(owner_id);
            }
          }
        }
      } // end for loop 
//       std::cout << "Recving from: " << recv_set.size() << " nodes." << std::endl;

      // initiate a series of nonblocking sends
      std::list< std::string > send_buffers;
      std::list<MPI::Request> send_requests;
      typedef send_map_type::value_type send_map_pair;
      foreach(send_map_pair& pair, send_map){
        // Deconstruct the pair
        owner_id_type owner_id = pair.first;
        const std::set<vertex_id_type>& send_set = pair.second;
        // initialize a string buffer and an archive
        std::stringstream strm;
        boost::archive::text_oarchive arc(strm);
        foreach(vertex_id_type v_id, send_set) {
          // serialize all bp messages
          serialize_bp_messages(arc, v_id);
        }
        send_buffers.push_back(strm.str());
        std::string& out_str = send_buffers.back();
        // Do the actual send
        MPI::Request request = 
          MPI::COMM_WORLD.Isend(out_str.c_str(),
                                out_str.size(),
                                MPI::CHAR,
                                owner_id,
                                BP_MESSAGE_TYPE);
        // save the request
        send_requests.push_back(request);
      } // end of sends

      // initiate a series of blocking recieves 
      foreach(owner_id_type owner, recv_set) {
        // Determine how much to receive from that owner
        MPI::Status status;
        MPI::COMM_WORLD.Probe(owner, BP_MESSAGE_TYPE, status);
        // Allocate a buffer
        buffer_type buffer;
        buffer.alloc(status.Get_count(MPI::CHAR));
        // Initiate a blocking recieve for that owner
        MPI::COMM_WORLD.Recv(buffer.body,
                             buffer.size,
                             MPI::CHAR,
                             owner,
                             BP_MESSAGE_TYPE);                             
        // Deserialize the message
        boost::iostreams::stream<boost::iostreams::array_source> 
          strm(buffer.body, buffer.size);
        boost::archive::text_iarchive arc(strm);
        deserialize_bp_messages(arc, strm);
        // Free the buffer
        buffer.free();
      } // end of blocking receives 

      // wait for all the sends to terminate
      foreach(MPI::Request& request, send_requests) {
        request.Wait();
      }
    } // end of exchange messages
    


    //! transfer messages from state manager into archive
    template<typename Archive>
    void serialize_bp_messages(Archive& arc, vertex_id_type v_id) {
      vertex_type v = factor_graph_->id2vertex(v_id);
      foreach(vertex_type u, factor_graph_->neighbors(v)) {
        vertex_id_type u_id = factor_graph_->vertex2id(u);
        message_type& msg = message(u,v);
        arc << u_id;
        arc << v_id;
        msg.save_remap(arc, var2id_);
      }
    } // end of serialize_bp_messages



    //! Load the message direclty into the state manager
    template<typename Archive, typename Stream>
    void deserialize_bp_messages(Archive& arc, Stream& strm) {
      // This may not work?  run until the stream is empty?
      while(strm.good()) {
        vertex_id_type src_id, dest_id;
        arc >> src_id;  
        arc >> dest_id;
        // Get the actual vertex for source and dest (in the local
        // representation)
        vertex_type src, dest;
        src = factor_graph_->id2vertex(src_id);
        dest = factor_graph_->id2vertex(dest_id);
        // Save the new message
        message_type new_msg;
        new_msg.load_remap(arc, id2var_);
        // Actually save the new message updating the schedule along
        // the way and doing any damping
        update_message(src, dest, new_msg);
      }
    } // end of deserialize_bp_messages



    
    /**
     * Send the beliefs to the root node 
     */
    void send_beliefs() {
      assert(factor_graph_ != NULL);
      assert(id() != MPI_ROOT_NODE);

      // Serialize all the beliefs
      std::stringstream strm;
      boost::archive::text_oarchive arc(strm);
      // Foreach vertex in the interior for this chunk serialize the
      // beliefs
      chunk_type& chunk = chunks_[id()];
      foreach(vertex_id_type v_id, chunk.interior) {
        vertex_type v = factor_graph_->id2vertex(v_id);
        if(v.is_variable()) {
          // get the belief
          const belief_type& blf = belief(v);
          // do serialization
          arc << v_id;
          blf.save_remap(arc, var2id_);
        }
      }

      // Send the size of the buffer
      int local_size = strm.str().size();
      MPI::COMM_WORLD.Gather(&local_size, 1, MPI::INT,
                             NULL, 1, MPI::INT,
                             MPI_ROOT_NODE);
      // Send the beliefs
      MPI::COMM_WORLD.Gatherv(strm.str().c_str(), strm.str().size(), MPI::CHAR,
                              NULL, NULL, NULL, MPI::CHAR,
                              MPI_ROOT_NODE);

    } // end of send beliefs






    /**
     * Receives the beliefs at the root node and incorporate into
     * local state manager
     */
    void recv_beliefs() {
      assert(factor_graph_ != NULL);
      assert(id() == MPI_ROOT_NODE);

      // Gather the ammount of data stored on each node including this one
      // currently this node does not send anything to its self.
      int local_size = 0;
      int* buffer_sizes = new int[num_processes()];
      // Gather the sizes of the belief buffers on each node
      MPI::COMM_WORLD.Gather(&local_size, 1, MPI::INT,
                             buffer_sizes, 1, MPI::INT,
                             MPI_ROOT_NODE);
      // assert that buffer_sizes contains the ammount of data each
      // node will send Construct the offsets
      // the offsets are the starting point for each proccess archive
      int* offsets = new int[num_processes()];
      offsets[0] = 0;
      size_t recv_buffer_size = buffer_sizes[0];
      for(size_t i = 1; i < num_processes(); ++i) {
        offsets[i] = offsets[i-1] + buffer_sizes[i-1];
        recv_buffer_size += buffer_sizes[i];
      }
      char* recv_buffer = new char[recv_buffer_size];
      // Collect all the beliefs
      MPI::COMM_WORLD.Gatherv(NULL, local_size, MPI::CHAR,
                              recv_buffer, buffer_sizes, offsets, MPI::CHAR,
                              MPI_ROOT_NODE);

      // Incorporate all the beliefs
      for(size_t i = 1; i < num_processes(); ++i) {
        // Deserialize and save all the beliefs
        boost::iostreams::stream<boost::iostreams::array_source> 
          strm((recv_buffer + offsets[i]), buffer_sizes[i]);
        boost::archive::text_iarchive arc(strm);
        while(strm.good()) {
          // Deserialize the ids first
          vertex_id_type vertex_id;
          arc >> vertex_id;  
          // Get the local vertex corresponding to the id
          vertex_type vertex = factor_graph_->id2vertex(vertex_id);
          // Update the belief
          belief(vertex).load_remap(arc, id2var_);
        } // end of while loop
      } // end of for loop
      // Free all memory buffers
      if(buffer_sizes != NULL) delete [] buffer_sizes;
      if(offsets != NULL) delete [] offsets;
      if(recv_buffer != NULL) delete [] recv_buffer;
    } // end of collect beliefs




    ///////////////////////////////////////////////////////////////////////////
    // BP Code    
    /**
     * Splash to convergence
     */
    void splash_to_convergence() {
      assert(schedule_.size() > 0);
      size_t update_count = 0;
      while(schedule_.top().second > bound_) {
//         if((update_count++ % 1000) == 0) {
//           vertex_id_type vid = factor_graph_->vertex2id(schedule_.top().first);
//           std::cout << vid << ", " << schedule_.top().second
//                     << std::endl;
//         }
        splash_once(schedule_.top().first);
      }
    }

    /**
     * Given a vertex this computes a single splash around that vertex
     */
    void splash_once(const vertex_type& root) {
      // Grow a splash ordering
      generate_splash(root);          
      // Push belief from the leaves to the root
      //-------------------------------------------------------------------
      revforeach(const vertex_type& v, splash_order_) {
        send_messages(v);
      }
      // Push belief from the root to the leaves (skipping the
      // root which was processed in the previous pass)
      //------------------------------------------------------------------
      foreach(const vertex_type& v, 
              std::make_pair(++splash_order_.begin(), splash_order_.end())) {
        send_messages(v);
      }
    } // End of splash_once


    /**
     * This function computes the splash ordering (a BFS) search for
     * the root vertex
     */
    void generate_splash(const vertex_type& root) {
      typedef std::set<vertex_type> visited_type;
      typedef std::list<vertex_type> queue_type;
      // Create a set to track the vertices visited in the traversal
      visited_type visited;
      queue_type splash_queue;
      // clear the global variables.
      splash_order_.clear();

      // Set the root to be visited and the first element in the queue
      splash_queue.push_back(root);
      visited.insert(root);
      // Grow a breath first search tree around the root      
      for(size_t i = 0; i < splash_size_ && !splash_queue.empty(); ++i) {
        // Remove the first element
        vertex_type u = splash_queue.front();
        splash_queue.pop_front();
        // Insert the first element into the tree order
        splash_order_.push_back(u);
        // If we need more vertices then grow out more
        if(splash_order_.size() + splash_queue.size() < splash_size_) {
          // Add all the unvisited neighbors to the queue
          foreach(const vertex_type& v, factor_graph_->neighbors(u)) {
            if((visited.count(v) == 0) && is_local(v) && 
               residual(v) >= bound_) {      
              splash_queue.push_back(v);
              visited.insert(v);
            }
          } // end of for each neighbors
        } // End of if statement
      } // End of foor loop
    } // End of Generate Splash


    /**
     * returns the residual of a given vertex
     */ 
    double residual(const vertex_type& vertex) {
      if(schedule_.contains(vertex)) return schedule_.get(vertex);
      else return 0.0;
    }

    /**
     * Determines whether the given vertex is local to this node.
     */
    bool is_local(const vertex_type& vertex) {
      return schedule_.contains(vertex);
    } // end of is local

    /**
     * Get the message from source to target
     */
    inline message_type& message(const vertex_type& source, 
                                 const vertex_type& target) {
      message_type& msg = messages_[target][source];
      // If the message does not already exists initialize it to
      // uniform distribution.
      if(msg.arguments().empty()) {
        domain_type domain = make_domain(source.is_variable() ? 
                                         &(source.variable()) :
                                         &(target.variable()));
        msg = message_type(domain, 1.0).normalize();
      } // end of if statement
      return msg;
    } // end of message


    /**
     * This writes the new message into the place of the old message
     * and updates the scheduling queue and does any damping necessary
     */
    inline void update_message(const vertex_type& source,
                               const vertex_type& target,
                               const message_type& new_msg) {
      // Create a norm object (this should be lightweight) 
      norm_type norm;
      // Get the original message
      message_type& original_msg = message(source, target);
      // Compute the norm
      double new_residual = norm(new_msg, original_msg);
      // Update the residual
      if(residual(target) < new_residual )
        schedule_.update(target, new_residual);
      // Require that there be no zeros
      assert(new_msg.minimum() > 0.0);
      // Save the new message
      original_msg = new_msg;
    } // end of update_message

    /**
     * Returnt he belief for a given vertex
     */
    inline belief_type& belief(const vertex_type& vertex) {
      belief_type& blf =  beliefs_[vertex];
      if(blf.arguments().empty()) {
        blf = 1.0;
      }
      return blf;
    } // end of belief


    /**
     * Receive all messages into the vertex and compute all new
     * outbound messages.
     */
    inline void send_messages(const vertex_type& source) {
      // For each of the neighboring vertices (vertex_target) of
      // vertex compute the message from vertex to vertex_target
      foreach(const vertex_type& dest, 
              factor_graph_->neighbors(source)) {
        // if the destination is local then send it
        if(is_local(dest)){
          send_message(source, dest);
        }
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
      

      if(source.is_variable()) {
        // create a temporary message to store the result into
        domain_type domain = make_domain(source.is_variable() ? 
                                         &(source.variable()) :
                                         &(target.variable()));
        new_msg_ = message_type(domain, 1.0).normalize();
        // If the source was a factor we multiply in the factor potential
      } else {
        // Set the message equal to the factor.  This will increase
        // the size of the message and require an allocation.
        new_msg_ = source.factor();
      } 

      // For each of the neighbors of the vertex
      foreach(const vertex_type& other, 
              factor_graph_->neighbors(source)) {
        // if this is not the dest_v
        if(other != target) {          
          // Combine the in_msg with the destination factor
          new_msg_.combine_in( message(other, source), csr_.dot_op);
          //          // Here we normalize after each iteration for numerical
          //           // stability.  This could be very costly for large factors.
          //           new_msg.normalize();
        }
      }        
      // If this is a message from a factor to a variable then we
      // must marginalize out all variables except the the target
      // variable.  
      if(source.is_factor()) {
        new_msg_ = new_msg_.collapse(make_domain(&target.variable()), csr_.cross_op);
      }
      // Normalize the message
      new_msg_.normalize();
      // Damp messages form factors to variables
      if(target.is_variable()) {
        new_msg_ = weighted_update(new_msg_, 
                                   message(source, target), 
                                   damping_);
      }
      // Do the actual update
      update_message(source, target, new_msg_);
    } // end of send_message


    /**
     * Update the belief for a given vertex
     */
    inline void update_belief(const vertex_type& vertex) {
      // create a temporary message to store the result into
      belief_type& blf = belief(vertex);
      // For each of the neighbors of the vertex
      foreach(const vertex_type& other, 
              factor_graph_->neighbors(vertex)) {
        // Combine the in_msg with the destination factor
        blf.combine_in( message(other, vertex), csr_.dot_op);
        // Here we normalize after each iteration for numerical
        // stability.  This could be very costly for large factors.
       blf.normalize();
      }        
      // Normalize the message
      blf.normalize();
    } // end of send_message


    ///////////////////////////////////////////////////////////////////////
    // MPI TOOLS
    /**
     * get the id of this process
     */
    size_t id() {
      int id = MPI::COMM_WORLD.Get_rank();
      assert(id >= 0);
      return id;
    }

    /**
     * Get the number of processes
     */
    size_t num_processes() {
      int numprocs = MPI::COMM_WORLD.Get_size();
      assert(numprocs >= 1);
      return numprocs;
    }

    /**
     * broadcast the object to all nodes.  May only be called from root
     * must be matched by all other nodes with an mpi_recv_bcast
     */
    template<typename T>
    void mpi_send_bcast(const T& object) {
      assert(id() == MPI_ROOT_NODE);
      std::stringstream strm;
      boost::archive::text_oarchive arc(strm);
      arc << object;
      mpi_bcast(strm.str().c_str(), strm.str().size());
    } // end of mpi_send_bcast

    /**
     * Recv end of mpi_bcast 
     */
    template<typename T>
    void mpi_recv_bcast(T& object) {
      buffer_type buffer = mpi_bcast();
      // deserialize the factor graph and maps
      boost::iostreams::stream<boost::iostreams::array_source> 
        strm(buffer.body, buffer.size);
      boost::archive::text_iarchive arc(strm);
      arc >> object;
      buffer.free();
    } // end of mpi_recv_bcast


    /**
     * Underlying mpi_bcast call.  If input is null then it is recv end
     * if input is not null then it must be root.
     * Must free buffer after use
     */
    buffer_type mpi_bcast(const char* input = NULL, size_t size = 0) {
      buffer_type buffer;
      assert(id() == MPI_ROOT_NODE || (input == NULL && size == 0));
      if(input != NULL && size > 0) {
        // Broadcast the message to all other nodes using the mpi bcast
        char* body = const_cast<char*>(input);
        buffer.body = body;
        buffer.size = size;
        // This will block until the graph is transmitted
        // Transmit the size of the datastructure
        MPI::COMM_WORLD.Bcast(reinterpret_cast<char*>(&(buffer.size)),
                              sizeof(buffer.size), 
                              MPI::CHAR, 
                              MPI_ROOT_NODE);
        // Transmit the the data
        MPI::COMM_WORLD.Bcast(buffer.body, buffer.size, 
                              MPI::CHAR, MPI_ROOT_NODE);
      } else { // Recieve data
        assert(id() != MPI_ROOT_NODE);
        assert(input == NULL);
        assert(size == 0);
        // Recieve the data
        MPI::COMM_WORLD.Bcast(reinterpret_cast<char*>(&(buffer.size)),
                              sizeof(buffer.size), 
                              MPI::CHAR, 
                              MPI_ROOT_NODE);
        assert(buffer.size > 0);
        // allocate enough space to store the body
        buffer.body = new char[buffer.size];
        // Get the body
        MPI::COMM_WORLD.Bcast(buffer.body, buffer.size, 
                              MPI::CHAR, MPI_ROOT_NODE);
      }
      return buffer;
    } // end of mpi_bcast

//     /**
//      * Reduce functions
//      */
//     void mpi_send_reduce(const char* input = NULL, size_t size = 0) {
//       assert(id() != MPI_ROOT_NODE);
//       // Send the size of the buffer
//       MPI::COMM_WORLD.Gather(&size, 1, MPI::INT,
//                              NULL, 1, MPI::INT,
//                              MPI_ROOT_NODE);
//       // Send the beliefs
//       MPI::COMM_WORLD.Gatherv(input, size, MPI::CHAR,
//                               NULL, NULL, NULL, MPI::CHAR,
//                               MPI_ROOT_NODE);
//     }




  }; // End of class strict map_reduce engine






}; // end of namespace
#include <sill/macros_undef.hpp>

#endif
















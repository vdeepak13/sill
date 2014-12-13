#ifndef MPI_GIBBS_ADAPTER_HPP
#define MPI_GIBBS_ADAPTER_HPP

// STL includes
#include <map>
#include <set>
#include <vector>
#include <ostream>
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
#include <sill/factor/util/norms.hpp>
#include <sill/math/gdl_enum.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/datastructure/mutable_queue.hpp>


// Should eventually define these elsewhere


// This include should always be last
#include <sill/macros_def.hpp>
namespace sill {

  /**
   * This adapter works with the mpi_engine
   */
  template<typename Engine>
  class mpi_gibbs_adapter {
    ///////////////////////////////////////////////////////////////////////
    // typedefs
  public:
    enum nodes {ROOT_NODE = 0};
    enum msg_types {BP_MESSAGE_TYPE = 0, TOKEN_TYPE = 1};

    typedef Engine engine_type;
    typedef typename engine_type::factor_type factor_type;
    typedef typename engine_type::belief_type belief_type;
    typedef typename engine_type::factor_graph_type factor_graph_type;
    typedef typename engine_type::vertex_set_type vertex_set_type;

    typedef typename factor_type::domain_type domain_type;
    typedef typename factor_graph_type::variable_type    variable_type;
    typedef typename factor_graph_type::vertex_type      vertex_type;
    typedef typename factor_graph_type::vertex_id_type   vertex_id_type;

    typedef uint32_t owner_id_type;
    typedef uint32_t variable_id_type;

    typedef typename engine_type::belief_map_type belief_map_type;
   


    /** simple buffer for storing character arrays in C style */
    class buffer_type { 
    public:
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
	try{
	  body = new char[size];
	} catch(...) {
	  std::cout << "Can't allocate: " << size << std::endl;
	  assert(false);
	}
      }
    };

    //! a map from the local variable pointer to its unique id (number)
    typedef std::map<variable_type*, variable_id_type> var2id_map_type;

    //! a map from the variables unique id to its local pointer
    typedef std::vector<variable_type*> id2var_map_type;


    
    ///////////////////////////////////////////////////////////////////////
    // Data members
  private:    
    //! factor graph
    factor_graph_type* factor_graph_;

    //! a map from the local variable pointer to its unique id (number)
    var2id_map_type var2id_;

    //! a map from the variables unique id to its local pointer
    id2var_map_type id2var_;

  public:
 
    /**
     * Create a map reduce engine.  This should be invoked on all nodes.
     */
    mpi_gibbs_adapter() { }

    ///////////////////////////////////////////////////////////////////////
    // Factor graph sending and receiving code
    /**
     * This is invoked on the root node at startup. The argument
     * provides the source factor graph on which inference is to be
     * run.
     */
    void send_factor_graph(factor_graph_type* factor_graph) {
      assert(id() == ROOT_NODE);
      assert(factor_graph != NULL);
      factor_graph_ = factor_graph;
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
     * startup.  This function ALLOCATES A NEW FACTOR GRAPH.  The
     * caller is responsible for destroying the factor graph after
     * execution.
     */
    factor_graph_type* recv_factor_graph() {
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
      assert(factor_graph_ != NULL);
      arc >> *factor_graph_;
      // Resurect the mappings
      arc >> var2id_;
      arc >> id2var_;
      // Delete the body data
      buffer.free();
      return factor_graph_;
    } // end of init called from not root

    
    ///////////////////////////////////////////////////////////////////////////
    // Sending and receiving beliefs code
    //////
    void sync_beliefs(belief_map_type& beliefs) {
      if(id() == ROOT_NODE) {
        recv_beliefs(beliefs);
      } else {
        send_beliefs(beliefs);
      }
    } // end of sync beliefs

    /**
     * Send the beliefs to the root node 
     */
    void send_beliefs(belief_map_type& beliefs) {
      // Serialize all the beliefs
      std::stringstream strm;
      boost::archive::text_oarchive arc(strm);
      // For each belief in the map
      typedef typename belief_map_type::value_type pair_type;
      foreach(pair_type& pair, beliefs) {
        vertex_type v = pair.first;
        vertex_id_type v_id = factor_graph_->vertex2id(v);
        belief_type& blf = pair.second;
        // do serialization
        arc << v_id;
        blf.save_remap(arc, var2id_);
      }
      // Send the size of the buffer
      int local_size = strm.str().size();
      MPI::COMM_WORLD.Gather(&local_size, 1, MPI::INT,
                             NULL, 1, MPI::INT,
                             ROOT_NODE);
      // Send the beliefs
      MPI::COMM_WORLD.Gatherv(strm.str().c_str(), strm.str().size(), MPI::CHAR,
                              NULL, NULL, NULL, MPI::CHAR,
                              ROOT_NODE);

    } // end of send beliefs

    /**
     * Receives the beliefs at the root node and incorporate into
     * local state manager
     */
    void recv_beliefs(belief_map_type& beliefs) {
      assert(factor_graph_ != NULL);
      assert(id() == ROOT_NODE);
      // Gather the ammount of data stored on each node including this one
      // currently this node does not send anything to its self.
      int local_size = 0;
      int* buffer_sizes = new int[num_processes()];
      // Gather the sizes of the belief buffers on each node
      MPI::COMM_WORLD.Gather(&local_size, 1, MPI::INT,
                             buffer_sizes, 1, MPI::INT,
                             ROOT_NODE);
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
                              ROOT_NODE);

      // Incorporate all the beliefs
      for(size_t i = 1; i < num_processes(); ++i) {
        // Deserialize and save all the beliefs
        boost::iostreams::stream<boost::iostreams::array_source> 
          strm((recv_buffer + offsets[i]), buffer_sizes[i]);
        boost::archive::text_iarchive arc(strm);
        while(strm.good()) {
          // Deserialize the ids first
          vertex_id_type v_id;
          arc >> v_id;  
          // Get the local vertex corresponding to the id
          vertex_type v = factor_graph_->id2vertex(v_id);
          belief_type blf;
          blf.load_remap(arc,id2var_);
          // Update the belief
          beliefs[v].combine_in(blf, sum_op);
        } // end of while loop
      } // end of for loop
      // Free all memory buffers
      if(buffer_sizes != NULL) delete [] buffer_sizes;
      if(offsets != NULL) delete [] offsets;
      if(recv_buffer != NULL) delete [] recv_buffer;
    } // end of collect beliefs



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
      assert(id() == ROOT_NODE);
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
      assert(id() == ROOT_NODE || (input == NULL && size == 0));
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
                              ROOT_NODE);
        // Transmit the the data
        MPI::COMM_WORLD.Bcast(buffer.body, buffer.size, 
                              MPI::CHAR, ROOT_NODE);
      } else { // Recieve data
        assert(id() != ROOT_NODE);
        assert(input == NULL);
        assert(size == 0);
        // Recieve the data
        MPI::COMM_WORLD.Bcast(reinterpret_cast<char*>(&(buffer.size)),
                              sizeof(buffer.size), 
                              MPI::CHAR, 
                              ROOT_NODE);
        assert(buffer.size > 0);
        // allocate enough space to store the body
        buffer.body = new char[buffer.size];
        // Get the body
        MPI::COMM_WORLD.Bcast(buffer.body, buffer.size, 
                              MPI::CHAR, ROOT_NODE);
      }
      return buffer;
    } // end of mpi_bcast
  }; // End of class mpi_gibbs_adapter


}; // end of namespace
#include <sill/macros_undef.hpp>

#endif
















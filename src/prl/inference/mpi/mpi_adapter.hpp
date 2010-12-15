#ifndef MPI_ADAPTER_HPP
#define MPI_ADAPTER_HPP

// STL includes
#include <map>
#include <set>
#include <vector>
#include <ostream>
#include <iostream>
#include <sstream>
#include <limits>
#include <queue>
#include <algorithm>

#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>

// MPI Libraries
#include <mpi.h>

// PRL Includes
#include <prl/model/factor_graph_model.hpp>
#include <prl/factor/table_factor.hpp>
#include <prl/mpi/mpi_wrapper.hpp>
#include <prl/parallel/pthread_tools.hpp>
#include <prl/model/factor_graph_partitioning.hpp>
#include <prl/serialization/serialize.hpp>
#include <prl/mpi/mpi_consensus.hpp>

// Should eventually define these elsewhere


// This include should always be last
#include <prl/macros_def.hpp>
namespace prl {

  
  /**
  * Callback to handle incoming BP messages.
  */
  class mpi_adapter_callback: public mpi_post_office::po_box_callback {
    public:
      
      std::queue<mpi_post_office::message> recvqueue;
      spinlock m;
      mpi_consensus& cons;

      mpi_adapter_callback(mpi_consensus &c):cons(c){ }
      
      void virtual recv_message(const mpi_post_office::message& msg) {
        cons.begin_critical_section();
        m.lock();
        recvqueue.push(msg.duplicate());
         m.unlock();
        cons.end_critical_section();
        cons.wakeup();
      }
  };
  

  /**
   * This adapter works with the mpi_engine2.
   * The adapter manages the transmission of messages and the ownership
   * of vertices. The Engine must have one public functions called
   *
   *  -- update_message_from_remote(vertex_type src, vertex_type dest, F &msg);
   *  -- root_update_belief(vertex_type v, F &msg);
   *
   * Incoming BP messages are buffered in a queue. When receive_bp_message is
   * called, it will pop BP messages from the queue and call
   * the Engine's update_message_from_remote function.
   *
   * root_update_belief will only be called on the root node. When inference is
   * complete, the root's Engine should call receive_beliefs, while nonroot
   * Engines should call send_beliefs_to_root. The receive_beliefs function will
   * call root_update_belief on each belief received.
   *
   */
  template<typename Engine>
  class mpi_adapter {
    private:
      ///////////////////////////////////////////////////////////////////////
      // typedefs
      typedef typename Engine::factor_type F;
      typedef typename Engine::model_type model_type;
      typedef size_t vertex_id_type;
      typedef typename model_type::vertex_type vertex_type;

      static const int MPI_BP_SYNC_PROTOCOL = 11;
      static const int MPI_BP_ASYNC_PROTOCOL = 12;
      // a private reference to the post office
      mpi_post_office &po_;
      mpi_consensus &cons_;
      mpi_adapter_callback bp_receiver_;
      universe *u_;
      size_t rootid_;
      model_type *rootmodel_;
      model_type localgraph_;
      // map from vertex to owner
      std::vector<size_t> globalvid2owner_;
      // now, this is actually unbelievably annoying, but since each
      // node's factor graph is different, each will have its own set of
      // "vertex id's". which will be different from node to node.
      // we will need to perform some remapping here.
      std::vector<size_t> local2globalvid_;
      std::vector<size_t> global2localvid_;

      size_t nummessagessent_;
      size_t numbytessent_;
      // for efficiency, we keep a set of all the vertices we own
      std::set<vertex_type> own_vertices_cache_;
      std::set<vertex_type> own_vertices_on_boundary_cache_;
     
      /**
        This struct stores the outgoing message queue going to a particular
        node. The queue will be flushed periodically based on various heuristics
        such as "age" and "change of message"
      */
      struct outgoing_queue{
        oarchive *arc;    /// the output archive
        std::stringstream *strm; /// the stream attached to the archive
        
        std::set<std::pair<size_t, size_t> > out_bp_messages;

        size_t age;       /// # calls to selective_flush
                          /// since this archive was lastflushed
        double urgency;   /// sum of change of message
      };
      
      std::vector<outgoing_queue> outqueues_;

      /// Fills the own_vertices_cache_ set
      void construct_own_vertices_cache() {
        own_vertices_cache_.clear();
        for (size_t i = 0;i < globalvid2owner_.size(); ++i) {
          if (globalvid2owner_[i] == po_.id()) {
            size_t localvid = global2localvid_[i];
            own_vertices_cache_.insert(localgraph_.id2vertex(localvid));
          }
        }

        own_vertices_on_boundary_cache_.clear();
        foreach (const vertex_type &v, own_vertices_cache_) {
          foreach(const vertex_type& other, localgraph_.neighbors(v)) {
            if (own_vertex(other) == false) {
              own_vertices_on_boundary_cache_.insert(v);
              break;
            }
          }
        }

        std::cout << "#own vertices " << own_vertices_cache_.size() << "\n";
      }
      /// Creates the outgoing serialization queues (one for each target MPI node
      void create_outgoing_queues() {
        outqueues_.resize(po_.num_processes());
        for (size_t i = 0; i < outqueues_.size(); ++i) {
          outqueues_[i].strm = new std::stringstream;
          outqueues_[i].arc = new oarchive(*(outqueues_[i].strm));
          outqueues_[i].out_bp_messages.clear();
          outqueues_[i].age = 0;
          outqueues_[i].urgency = 0.0;
        }
      }
      /// Fills the globalvid2owner_ datastructure from partitioning information
      void construct_globalvid2owner(model_type &rootmodel,
                                      factor_graph_partition<F> &part,
                                      std::vector<size_t> &assignments) {
        // keep a counter of the number of vertices a partion,
        // and fill the vertex to owner table
        std::vector<size_t> numverticesperproc;
        numverticesperproc.resize(po_.num_processes());
        globalvid2owner_.resize(rootmodel.num_vertices());
        
        foreach(vertex_type v, rootmodel.vertices()) {
          globalvid2owner_[rootmodel.vertex2id(v)] =
                        assignments[part.vertex2part(v)];
          numverticesperproc[assignments[part.vertex2part(v)]]++;
        }
        std::cout<<"vperproc=[";
        for (size_t i = 0; i < po_.num_processes(); ++i) {
          std::cout<<numverticesperproc[i]<<",";
        }
        std::cout<<"]";
      }


      /**
      'part' is a partitioning of the factor graph in 'rootmodel'
      'assignments' is a mapping from the partition number to the MPI node
      number.
      For each MPI node, this function serializes the subgraph belonging to it
      and sends it out via MPI. The remote node receives it by calling
      receive_factor_graph
      */
      void send_factor_graph(model_type &rootmodel,
                              factor_graph_partition<F> &part,
                              std::vector<size_t> &assignments) {
        /*
        - serialize all factors involved in its subgraph
        - annoyingly enough, to build the local2globalvid_
        and global2localvid_ mappings, we not just have to transmit the
        global vertex ids for the factors, but also the variables...

        The archive format is as such
        [universe]
        [globalvid2owner]
        ---- A list of factors prefixed with a boolean true ----
        [true]
        [factor vertex id]
        [factor]
        ---- The list of factors terminate with a boolean false ----
        [false]
        [# variables]
        ---- For each variable ----
        [variable vertex id]
        [variable*]
        */

        // not exactly the most efficient. TODO: otpimize
        for (size_t node = 0; node < po_.num_processes(); ++node) {
          // we must transmit the universe first.
          std::stringstream ss;
          oarchive oarc(ss);
          oarc << *u_;
          oarc << globalvid2owner_;

          std::set<vertex_type> variablesinvolved;
          for(size_t i = 0;i < globalvid2owner_.size(); ++i) {
            if (globalvid2owner_[i] != node) continue;
            vertex_type v = rootmodel.id2vertex(i);
            // if it is a factor add it. We need to transmid the global vid
            // of the factor as well to allow the other party to construct
            // the local2globalvid_ and global2localvid_ mappings
            if (v.is_factor()) {
              // serialize the id of the factor
              oarc << true;
              oarc << rootmodel.vertex2id(v);
              oarc << v.factor();
              // remmeber all the accompanying variables
              foreach (vertex_type n, rootmodel.neighbors(v)) {
                variablesinvolved.insert(n);
              }
            }
            else if (v.is_variable()){
              // remember the variable
              variablesinvolved.insert(v);
              // this is a variable, so I need to include all of its
              // neighboring factors which are on a different machines.
              foreach (vertex_type n, rootmodel.neighbors(v)) {
                if (n.is_factor() &&
                      globalvid2owner_[rootmodel.vertex2id(n)] != node) {
                  oarc << true;
                  oarc << rootmodel.vertex2id(n);
                  oarc << n.factor();
                }
              }
            }
          }

          oarc << false;
          oarc << variablesinvolved.size();
          foreach(vertex_type v, variablesinvolved) {
            oarc << rootmodel.vertex2id(v);
            oarc << &(v.variable());
          }
          // transmit
          po_.send_message(node, MPI_BP_SYNC_PROTOCOL,
                            ss.str().length(), ss.str().c_str());
        }
      }


      /**
      * This function reads the factor graph sent via send_factor_graph()
      * and constructs a local subgraph of the original factor graph.
      * This function will also build the necessary id mappings.
      * If "overwriteuniverse" is set (true by default)
      * the universe will be overwritten with the root's universe.
      * Set it to false if you already have a matching universe you would like
      * to use.
      */
      void receive_factor_graph(universe& u,
                                bool overwriteuniverse = true) {
        // save a pointer to the universe
        // read the message from the post office
        mpi_post_office::message msg;
        po_.receive(MPI_BP_SYNC_PROTOCOL, msg);
        rootid_ = msg.orig;
        // deserialize the stream
        boost::iostreams::stream<boost::iostreams::array_source>
                            strm(msg.body, msg.body_size);
        iarchive iarc(strm);
        // read the universe and attach it to the stream
        if (overwriteuniverse) {
          iarc >> u;
        }
        else {
          universe unused;
          iarc >> unused;
        }
        iarc.attach_universe(&u);
        globalvid2owner_.clear();
        iarc >> globalvid2owner_;

                // read the graph and build up the local2globalvid_
        // and global2localvid_ mapping

        localgraph_.clear();

        // read all the factors and update the vid mappings along the way
        while(strm.good()) {
          bool isfactor;
          F tablefactor;
          vertex_id_type globalvid;
          // check if we are done
          iarc >> isfactor;
          if (isfactor == false) break;
          iarc >> globalvid;
          iarc >> tablefactor;
          vertex_id_type localvid = localgraph_.add_factor(tablefactor);
          // resize the vid mapping to fit
          local2globalvid_.resize(std::max(local2globalvid_.size(), size_t(localvid+1)));
          global2localvid_.resize(std::max(global2localvid_.size(), size_t(globalvid+1)));
          local2globalvid_[localvid] = globalvid;
          global2localvid_[globalvid] = localvid;
        }
        size_t numvariables;
        iarc >> numvariables;
        localgraph_.rebuild_neighbors_and_indexes();
        // read all the variables and update the vid mappings
        for (size_t i = 0; i < numvariables; ++i) {
          vertex_id_type globalvid;
          typename model_type::variable_type *var;
          iarc >> globalvid;
          iarc >> var;
          // construct the vertex_type and ask the localgraph for the vertexid
          vertex_id_type localvid = localgraph_.to_vertex(var).id();
          // resize the vid mapping to fit
          local2globalvid_.resize(std::max(local2globalvid_.size(), size_t(localvid+1)));
          global2localvid_.resize(std::max(global2localvid_.size(), size_t(globalvid+1)));
          local2globalvid_[localvid] = globalvid;
          global2localvid_[globalvid] = localvid;
        }

        delete [] msg.body;
        //std::cout << global2localvid_ << "\n";
        //std::cout << local2globalvid_ << "\n";
        
      }

    public:

      mpi_adapter(mpi_post_office &po, mpi_consensus &cons):po_(po),
                                                             cons_(cons),
                                                             bp_receiver_(cons){
        rootmodel_ = NULL;
        bp_receiver_.m.lock();
        po_.register_handler(MPI_BP_SYNC_PROTOCOL, NULL);
        po_.register_handler(MPI_BP_ASYNC_PROTOCOL, &bp_receiver_);
        bp_receiver_.m.unlock();
        nummessagessent_ = 0;
      }
      ~mpi_adapter(){
        rootmodel_ = NULL;
        bp_receiver_.m.lock();
        po_.unregister_handler(MPI_BP_SYNC_PROTOCOL);
        po_.unregister_handler(MPI_BP_ASYNC_PROTOCOL);
        bp_receiver_.m.unlock();
      }

      void sync_send_all_but_self(const std::string& s) {
        for (size_t i = 0; i < po_.num_processes(); ++i) {
          if (i != po_.id()) {
            po_.send_message(i, MPI_BP_SYNC_PROTOCOL,
                            s.length(), s.c_str());
          }
        }
      }

      void sync_send(size_t targetmachine, std::string &s) {
        assert(targetmachine < po_.num_processes());
        po_.send_message(targetmachine, MPI_BP_SYNC_PROTOCOL,
                            s.length(), s.c_str());
      }

      std::string sync_recv() {
        mpi_post_office::message msg;
        po_.receive(MPI_BP_SYNC_PROTOCOL, msg);
        std::string ret(msg.body, msg.body_size);
        delete [] msg.body;
        return ret;
      }
      /**
      * Root node initializes the adapter by calling this function.
      * Other nodes should call initialize_nonroot() at the same time.
      *
      * The adapter read the partitioning and transmit the factor graph
      * segments to the other nodes.
      */
      void initialize_root(universe& u,
                          model_type &rootmodel,
                          factor_graph_partition<F> &part) {
        rootmodel_ = &rootmodel;
        // keep a pointer to the universe. we will need it in the future
        // for fast serialization
        u_ = &u;
        // randomly assign partitions to processes
        std::vector<size_t> assignments;
        for(size_t i = 0;i < part.number_of_parts();++i) {
          assignments.push_back(i % po_.num_processes());
        }
        std::random_shuffle(assignments.begin(),assignments.end());

        construct_globalvid2owner(rootmodel, part, assignments);
        // transmit factor graph
        send_factor_graph(rootmodel, part, assignments);
        // we call initialize_non_root as well to receive the data since
        // we transmit to self as well. Use our current universe.
        receive_factor_graph(*u_, false);
        
        construct_own_vertices_cache();
        create_outgoing_queues();
      }
      /**
      * initialization routine called by non-root nodes.
      */
      void initialize_nonroot(universe& u) {
        rootmodel_ = NULL;
        u_ = &u;
        receive_factor_graph(*u_);
        construct_own_vertices_cache();
        create_outgoing_queues();
      }

      
      model_type& get_local_graph() {
          return localgraph_;
      }
      
      bool own_vertex(const vertex_type &localvertex) {
        return own_vertices_cache_.count(localvertex) > 0;
      }
      bool own_vertex(const vertex_id_type &localvid) {
        return globalvid2owner_[local2globalvid_[localvid]] == po_.id();
      }
      bool on_boundary(const vertex_type &localvertex) {
        return own_vertices_on_boundary_cache_.count(localvertex) > 0;
      }
      bool on_boundary(const vertex_id_type &localvid) {
        return on_boundary(localgraph_.id2vertex(localvid));
      }


      vertex_id_type to_global_vid(vertex_id_type localvid) {
        return local2globalvid_[localvid];
      }
      
      vertex_id_type to_local_vid(vertex_id_type globalvid) {
        return global2localvid_[globalvid];
      }
      void queue_outgoing_bp_message(const vertex_type &src,
                                     const vertex_type &dest,
                                     const F &message,
                                     double delta = 0.0) {
        vertex_id_type localdestvid = localgraph_.vertex2id(dest);
        vertex_id_type globaldestvid = local2globalvid_[localdestvid];

        vertex_id_type localsrcvid = localgraph_.vertex2id(src);
        vertex_id_type globalsrcvid = local2globalvid_[localsrcvid];
        size_t destinationnode = globalvid2owner_[globaldestvid];
        
        std::pair<size_t, size_t> srcdestkey(globalsrcvid, globaldestvid);
        
        outqueues_[destinationnode].out_bp_messages.insert(srcdestkey);
        
        outqueues_[destinationnode].urgency += delta;
      }

      void send_outgoing_bp_message(const vertex_type &src,
                                    const vertex_type &dest,
                                    const F &message) {
        vertex_id_type localdestvid = localgraph_.vertex2id(dest);
        vertex_id_type globaldestvid = local2globalvid_[localdestvid];

        vertex_id_type localsrcvid = localgraph_.vertex2id(src);
        vertex_id_type globalsrcvid = local2globalvid_[localsrcvid];
        size_t i = globalvid2owner_[globaldestvid];
        *(outqueues_[i].arc) 
                        << true
                        << globalsrcvid
                        << globaldestvid
                        << message;
        *(outqueues_[i].arc) << false;
         outqueues_[i]->flush();
         po_.send_message(i, MPI_BP_ASYNC_PROTOCOL,
                            outqueues_[i].strm->str().length(),
                            outqueues_[i].strm->str().c_str());
        outqueues_[i].strm->str("");
//        std::cout << outqueues_[i].strm->str().length() << "\t";

      }



      void flush_queue_to_stream(size_t i, Engine* engine) {
        typedef std::pair<size_t, size_t> srcdestkey_type;
        foreach(srcdestkey_type srcdestkey, outqueues_[i].out_bp_messages) {
          *(outqueues_[i].arc) 
                        << true
                        << srcdestkey.first
                        << srcdestkey.second
                        << engine->message(srcdestkey.first, srcdestkey.second);
        }
        outqueues_[i].out_bp_messages.clear();
      }


      
      void selective_flush(Engine* engine) {
//        flush(false);
//        return;
        // TODO
        for (size_t i = 0; i < outqueues_.size(); ++i) {
          if (outqueues_[i].age > 3) {
            flush_queue_to_stream(i, engine);
            *(outqueues_[i].arc) << false;
            po_.send_message(i, MPI_BP_ASYNC_PROTOCOL,
                              outqueues_[i].strm->str().length(),
                              outqueues_[i].strm->str().c_str());
            numbytessent_ += outqueues_[i].strm->str().length();
            nummessagessent_++;
//            delete outqueues_[i].arc;
//            delete outqueues_[i].strm;
//            outqueues_[i].strm = new std::stringstream;
//            outqueues_[i].arc = new oarchive(*(outqueues_[i].strm));
            outqueues_[i].strm->str("");
            outqueues_[i].age = 0;
            outqueues_[i].urgency = 0.0;
          }
          else {
            outqueues_[i].age++;
          }
        }
      }
      
      /**
       Sends all the pending outgoing messages
       if flushpobuffer is true, this will request a flush of the post
       office buffer as well (more time consuming)
      */
      void flush(Engine* engine, bool flushpobuffer = true) {
        for (size_t i = 0; i < outqueues_.size(); ++i) {
          flush_queue_to_stream(i, engine);
          if (outqueues_[i].strm->str().length() > 0) {
            *(outqueues_[i].arc) << false;
            po_.send_message(i, MPI_BP_ASYNC_PROTOCOL,
                              outqueues_[i].strm->str().length(),
                              outqueues_[i].strm->str().c_str());
            numbytessent_ += outqueues_[i].strm->str().length();
            nummessagessent_++;
          //  delete outqueues_[i].arc;
//            delete outqueues_[i].strm;
//
            //outqueues_[i].strm = new std::stringstream;
//            outqueues_[i].arc = new oarchive(*(outqueues_[i].strm));
            outqueues_[i].strm->str("");
            outqueues_[i].age = 0;
            outqueues_[i].urgency = 0.0;
          }
        }
        if (flushpobuffer) {
          po_.flush();
        }
      }

      double receive_bp_messages(Engine* engine) {
        size_t numrecv = 0;
        // lock the queue to pop the message
        bp_receiver_.m.lock();
        while (!bp_receiver_.recvqueue.empty()) {
          mpi_post_office::message msg = bp_receiver_.recvqueue.front();
          
          bp_receiver_.recvqueue.pop();
          // unlock it while I process the message
          bp_receiver_.m.unlock();

          // create the archive
          boost::iostreams::stream<boost::iostreams::array_source>
                           strm(msg.body, msg.body_size);
          iarchive arc(strm);
          arc.attach_universe(u_);
          while (strm.good()) {
            bool hasnext;
            arc >> hasnext;
            if (hasnext == false) break;
            // deserialize the stream
            vertex_id_type localdestvid, globaldestvid;
            vertex_id_type localsrcvid, globalsrcvid;
            F message;
            arc >> globalsrcvid >> globaldestvid >> message;
            localsrcvid = global2localvid_[globalsrcvid];
            localdestvid = global2localvid_[globaldestvid];

            engine->update_message_from_remote(localgraph_.id2vertex(localsrcvid),
                                      localgraph_.id2vertex(localdestvid),
                                      message);
            ++numrecv;
          }
          bp_receiver_.m.lock();
          delete [] msg.body;
        }
        bp_receiver_.m.unlock();
        return numrecv;
      }
      /**
      * Takes in the beliefs of the local model and transmits it to the root
      */
      void send_beliefs_to_root(std::map<vertex_type, F> &beliefs) {
        //create the stream
        std::stringstream ss;
        oarchive arc(ss);

        // put in all the beliefs of my own vertices
        foreach (const vertex_type &v, localgraph_.vertices()) {
          if (own_vertex(v)) {
            arc << true;
            arc << local2globalvid_[localgraph_.vertex2id(v)];
            arc << beliefs[v];
          }
        }
        arc << false;
        // send it to node 0
        po_.send_message(rootid_, MPI_BP_SYNC_PROTOCOL,
                          ss.str().length(), ss.str().c_str());
      }
      
      void send_beliefs_to_root(std::vector<F> &beliefs) {
        //create the stream
        std::stringstream ss;
        oarchive arc(ss);

        // put in all the beliefs of my own vertices
        foreach (const vertex_type &v, localgraph_.vertices()) {
          if (own_vertex(v)) {
            arc << true;
            arc << local2globalvid_[localgraph_.vertex2id(v)];
            arc << beliefs[v.id()];
          }
        }
        arc << false;
        // send it to node 0
        po_.send_message(rootid_, MPI_BP_SYNC_PROTOCOL,
                          ss.str().length(), ss.str().c_str());
      }

      /**
      * Receives the beliefs from each node's local model and integrates
      * it into the root model. Since the root node also does computation,
      * this function therefore also takes in the localbeliefs as an argument
      */
      void root_receive_beliefs(Engine* engine) {
        assert(rootmodel_ != NULL);

        int count = 0;
        std::set<size_t> received_from_node;
        //create the stream
        while(received_from_node.size() != po_.num_processes()) {
          mpi_post_office::message msg;
          po_.receive(MPI_BP_SYNC_PROTOCOL, msg);
          received_from_node.insert(msg.orig);
          // deserialize the stream
          boost::iostreams::stream<boost::iostreams::array_source>
          strm(msg.body, msg.body_size);
          iarchive iarc(strm);
          iarc.attach_universe(u_);
          // put in all the beliefs of my own vertices
          while(1) {
            bool hasnext;
            iarc >> hasnext;
            if (hasnext == false) break;
            vertex_id_type globalvid;
            F belief;
            iarc >> globalvid;
            iarc >> belief;
            engine->root_update_belief(rootmodel_->id2vertex(globalvid), belief);
            count++;
          }
        }
      }
      
      size_t get_rootid(){
        return rootid_;
      }

      size_t nummessagessent() {
      return nummessagessent_;
      }
      
      size_t numbytessent() {
      return numbytessent_;
      }

      void clearstats() {
        nummessagessent_ = 0;
        numbytessent_ = 0;
      }
  }; // End of class mpi_adapter_threaded


}; // end of namespace
#include <prl/macros_undef.hpp>

#endif // MPI_ADAPTER2

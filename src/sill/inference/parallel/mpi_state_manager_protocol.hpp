#ifndef MPI_STATE_MANAGER_PROTOCOL_HPP
#define MPI_STATE_MANAGER_PROTOCOL_HPP

// #define DEBUG
// forward declaration
namespace sill {
template <typename F>
class mpi_state_manager;
}


#include <iostream>
#include <sstream>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>

#include <sill/mpi/mpi_wrapper.hpp>
#include <sill/inference/parallel/mpi_state_manager.hpp>
#include <sill/parallel/pthread_tools.hpp>

#include <sill/macros_def.hpp>
namespace sill {

/**
  Design Specs:
  
  You should read the design spec in mpi_state_manager first.
  
  The design here is relatively simple. We wrap an additional "message type"
  (MTYPE) header which takes 4 bytes into the front of a message.
  
  Upon receiving a message through the recv_message() function, the header is
  stripped and passed into the parse_message() function. the parse_message() 
  function is just a switch statement over the possible message types. 
  
  Sending a message is similar. send_message(message_type) does the work
  for the simpler messages. It will dig into the mpi_state_manager class
  if needed. Therefore if synchronization is necessary, parse_message and
  send_message (and variants) will have to be updated.
  
  There are a few specialized send_message variants that take additional 
  arguments such as send_message_update(), which sends a BP message update to
  a remote node; and give_away_vertex(), which removes the vertex from the
  local set of messages and gives it away. 
  
  Message Types:
  All messages types are defined by a 4 byte int32_t
  Some messages, when sent, will wait for an "acknowledge" from the remote
  parties before continuing execution. These messages are denoted by having
  the message value being negative. The ACK message itself has value 0.
  We call the messages which require acknowledgement, "blocking messages"
  
  MTYPE_PARAMETERS: will send epsilon and max_vertices. The receiver will
                    overwrite the mpi_state_manager's value with the received
                    values
  MTYPE_FACTOR_GRAPH: will send the factor graph model, a complete owner<->vertex
                      mapping, and a complete vertex<->id mapping.
                      As again, receiver will overwrite its mpi_state_manager
                      with the received data
  MTYPE_SYN:  Does nothing but just waits for a reply (unused as yet)
  MTYPE_BP_MESSAGE: Sends a message update. Receiver will integrate the update
                    (and compute the new residual). If receiver does not own
                    this message, this message is forwarded to who it thinks
                    owns the message using its own owner<->vertex mapping
  MTYPE_VERTEX_TRANSFER: transfers ownership of a vertex to another node (not implemented)
  MTYPE_VERTEX_OWNER_UPDATE: transfers a single vertexid->ownerid. Receiver
                             will update its owner<->vertex mappings with the
                             new value (not implemented)

  Now, if a VERTEX_TRANSFER occurs, it may be reasonable that the sender or
  receiver broadcasts a VERTEX_OWNER_UPDATE. But this is extraordinarily
  expensive if we are to do it for every vertex. An alternative is the 
  observation if a vertex transfer occurs, only that the neighbors of that vertex
  are affected. Therefore, we only need to inform the nodes who owns its neighbors.
  
  If a distant BP_MESSAGE is issued for whatever reason, the forwarding behavior
  is guaranteed to eventually reach the node who actually owns the message. 
  
  Yet another alternative is a lazy combination of the above 2 ideas. If a 
  VERTEX_TRANSFER occurs, only the source and destination node update their own
  vertex<->owner mappings. Then if any node receives a BP_MESSAGE that has to
  be forwarded, the node will forward the message, and simultaneously send a 
  VERTEX_OWNER_UPDATE to the sender (therefore if the sender has to send any
  more updates, it will go to the right node immediately)
  
  Message Acknowledgement method:
  We note that there is no message ID system. Therefore acknowledgement
  simply works by an acknowledge counter. Therefore if this node broadcasts to
  10 nodes, it will block in the send() function until it receives any 10 ACKS.
  
  There is a small problem with overlapping ACKS; as in if I call send() to 10 
  nodes, then with another thread call send() to another 10 nodes, the ACKS to 
  the different messages will overlap. To avoid this, only 1 blocking message
  is allowed to enter send at any one time. (protection with a condition
  variable). However, you are allowed to have 1 blocking message waiting in
  "send", and at the same time keep sending other non-blocking messages.
*/
namespace mpi_state_manager_prot {
  // protocol number
  const int MPI_STATE_MANAGER_PROT_ID = 128;
  
  // message types
  // All messages with numbers < 0 are blocking 
  // basically it will wait for an ack from the otherside before continuing
  const int32_t MTYPE_ACK = 0;  
  const int32_t MTYPE_PARAMETERS = -1;    // sends the state manager parameters
  const int32_t MTYPE_FACTOR_GRAPH = -2; // sends the graphical model
  const int32_t MTYPE_SYN = -3;   // does nothing. except for wait for a reply
  
  const int32_t MTYPE_BP_MESSAGE = 1; // TO TEST: sends a message containing a BP message
  const int32_t MTYPE_VERTEX_TRANSFER = 2; // TODO: sends a message containing a vertex
  const int32_t MTYPE_OWNER_UPDATE = 3; //TODO:

  const int32_t MTYPE_FINISHED = 4;
  const int32_t MTYPE_RESUMED = 5;
  const int32_t MTYPE_SHUTDOWN = 6;
  const int32_t MTYPE_BELIEF = 7;
  const int32_t MTYPE_SEND_BELIEF = -4;  

  /**
  The protocol handler for the mpi_state_manager
  */
  template <typename F>
  class mpi_state_manager_protocol
                        : public mpi_post_office::po_box_callback {
   private:
  
    mpi_state_manager<F> &state_;
    mpi_post_office &po_;
    
    mutex blocking_messages_;

    int ackcount_;
    mutex ackmutex_;
    conditional ackcond_;
    bool finished_;
    
    int messages_sent;
    int vertices_transfered;
    int vertices_received;

   public:
    
    mpi_state_manager_protocol(mpi_state_manager<F> &state, mpi_post_office &po)
                              :state_(state), po_(po), ackcount_(0), 
                               finished_(false), messages_sent(0), 
                               vertices_transfered(0), vertices_received(0){
    }
    
  
    // receives an MPI message
    void recv_message(const mpi_post_office::message& msg) {
      // get the type of the message
      int32_t typeofmessage = *(reinterpret_cast<const int32_t*>(msg.body));
      #ifdef DEBUG
      std::cout << "Received Message Type: " << typeofmessage << std::endl;
      #endif
      // put the rest of the message into a serialization archive
      const char* body = msg.body + sizeof(int32_t);
      size_t bodylen = msg.body_size - sizeof(int32_t);
      // under ocassions (such as an ACK) there is no body
      // then the 'archive' gets upset. 
      if (bodylen > 0) {
        boost::iostreams::stream<boost::iostreams::array_source> strm(body, bodylen);
        boost::archive::binary_iarchive arc(strm);
        parse_message(typeofmessage, &arc, msg.body, msg.body_size, msg.orig);
      }
      else {
        parse_message(typeofmessage, NULL, msg.body, msg.body_size, msg.orig);
      }
      // process the message
      
    }
    
    /**
      Parses the message. both the serialization archive (arc) and the raw
      contents (as a const char*) are passed in since some applications might 
      find it easier to use the raw data. 
      
      if the body is of 0 length, arc will be NULL as the archive gets upset if 
      you try to create it with an empty string
    */
    void parse_message(int32_t typeofmessage, 
                      boost::archive::binary_iarchive *arc, 
                      const char* wholemessage, 
                      size_t msglen,
                      int src) {
      switch (typeofmessage) {
        // its an ACK. I need to increment the ack counter
        // and signal since the 'sender' will be waiting for this
        case MTYPE_ACK:
          ackmutex_.lock();
          ackcount_++;
          ackcond_.signal();
          ackmutex_.unlock();
          break;
          
        // deserialize the parameters
        case MTYPE_PARAMETERS:
          (*arc) >> state_.epsilon_;
          (*arc) >> state_.max_vertices_;
          break;
          
        // deserialize the factor graph
        case MTYPE_FACTOR_GRAPH:
          assert(state_.model_ == NULL);
          state_.model_ = new typename mpi_state_manager<F>::factor_graph_model_type;
          (*arc) >> *(state_.model_);
          (*arc) >> state_.vertex2owner_;
          (*arc) >> state_.owner2vertex_;
          (*arc) >> state_.var2id_;
          (*arc) >> state_.id2var_;
          state_.ConstructLocalMessages();
          state_.CreateSchedule();
          break;

        case MTYPE_BP_MESSAGE: {
            // if I am not the recipient of this message. forward it
            // this is where the char* version of the body is useful :-)
            
            // read the vertex ids
            uint32_t srcid; 
            uint32_t destid;
            (*arc) >> srcid;
            (*arc) >> destid;
            #ifdef DEBUG
            std::cout << "BP msg from " << srcid << " to " << destid << std::endl;
            #endif
            // check the destination vertex
            typename mpi_state_manager<F>::vertex_type destv = 
                                                state_.model_->id2vertex(destid);
            // see if I own this vertex
            uint32_t destination = state_.vertex2owner_[destv]; 
            // I do not!
            // send it back out
            if (destination != po_.id()) {
              // see if it is in the deferred list
              state_.deferred_insertions_lock.lock();
              bool isdeferred = false;
              foreach(typename mpi_state_manager<F>::deferred_insertion_data &p, 
                      state_.deferred_insertions) {
                if (p.v == destv) {
                  isdeferred = true;
                  break;
                }
              }
              if (isdeferred) {
                typename mpi_state_manager<F>::edge_type e(state_.model_->id2vertex(srcid), 
                                                           destv);
                state_.deferred_receives[e].load_remap(*arc, state_.id2var_);
              }
              state_.deferred_insertions_lock.unlock();

              if (isdeferred) {
                state_.shutdown_cond_.broadcast();
                state_.finishedcounter_cond_.broadcast();
                break;
              }
              
              
              po_.send_message(destination, 
                              MPI_STATE_MANAGER_PROT_ID, 
                              msglen,
                              wholemessage);
              //#ifdef DEBUG
              // std::cout << "forward" << std::endl;
              //#endif
              std::stringstream strm;
              strm.write(reinterpret_cast<const char*>(&MTYPE_OWNER_UPDATE), sizeof(int32_t));
              boost::archive::binary_oarchive outarc(strm);
              outarc << destid;
              outarc << destination;
              po_.send_message(src,
                              MPI_STATE_MANAGER_PROT_ID, 
                              strm.str().length(), 
                              strm.str().c_str());

              // TODO: and send an owner update to src
              // 
            }
            else {
            typename mpi_state_manager<F>::vertex_type srcv = 
                                                state_.model_->id2vertex(srcid);
              // HACK:
              // handle receipt of message. how?
              // take advantage of the fact that this parse_message runs 
              // in a different thread as the residual BP
              // therefore I can safely check out a message for writing (yeah!)
              
              // check out the message
              typename mpi_state_manager<F>::message_type *msg
                                  = state_.checkout(srcv, destv, Writing);
              // deserialize
              msg->load_remap(*arc, state_.id2var_);
              state_.checkin(srcv, destv, msg);
              assert(msg->arg_vector().size() > 0);
              // if I receive a message, finish() should wake up if it 
              // is sleeping
              state_.shutdown_cond_.broadcast();
              // if I am root, I will be waiting on the finished counter
              // so lets broadcast to it too
              state_.finishedcounter_cond_.broadcast();
            }
            break;
        }
        
        case MTYPE_FINISHED:
          state_.finishedcounter_mut_.lock();
          state_.finishedcounter_++;
          state_.finishedcounter_cond_.broadcast();
          state_.finishedcounter_mut_.unlock();
          break;
          
        case MTYPE_RESUMED:
          state_.finishedcounter_mut_.lock();
          state_.finishedcounter_--;
          state_.finishedcounter_cond_.broadcast();
          state_.finishedcounter_mut_.unlock();
          break;
          
        case MTYPE_SHUTDOWN:
          state_.shutdown_mut_.lock();
          state_.shutdown_ = true;
          state_.shutdown_cond_.broadcast();
          state_.shutdown_mut_.unlock();
          break;
        case MTYPE_SEND_BELIEF:
          state_.SendBeliefs();
          break;
        case MTYPE_BELIEF: {
          uint32_t vid;
          (*arc) >> vid;
          #ifdef DEBUG
          std::cout << "BP belief at " << vid;
          #endif
          // check the destination vertex
          typename mpi_state_manager<F>::vertex_type v = 
                                              state_.model_->id2vertex(vid);
          typename mpi_state_manager<F>::belief_type* b = state_.checkout_belief(v);
          b->load_remap(*arc, state_.id2var_);
          state_.checkin_belief(v,b);
          break;
        }
        case MTYPE_OWNER_UPDATE: {
            uint32_t vid;
            uint32_t owner;
            (*arc) >> vid;
            (*arc) >> owner;
            // never up to anyone else to tell me what I know
            if (owner == po_.id() || 
                state_.vertex2owner_[state_.model_->id2vertex(vid)] == po_.id()) {
              // std::cout << "Conflicting owner knowledge! " << vid;
            }
            else {
              // std::cout << "Owner Update " << vid;
              state_.SetOwner(state_.model_->id2vertex(vid),owner);
            }
            break;
          }
        case MTYPE_VERTEX_TRANSFER: {
          uint32_t vid;
          double residual;
          typename mpi_state_manager<F>::deferred_insertion_data p;
          (*arc) >> vid;
          (*arc) >> residual;
          // std::cout << "Integrating vertex " << vid << std::endl;
          vertices_received++;
          typename mpi_state_manager<F>::vertex_type v = 
                                              state_.model_->id2vertex(vid);
          p.v = v;
          p.priority = residual;
          if (v.is_variable()) {
            p.belief.load_remap(*arc, state_.id2var_);
            assert(p.belief.arg_vector().size() > 0);
          }
          int numneighbors = state_.model_->num_neighbors(v);
          for (int i = 0;i < numneighbors; ++i) {
            uint32_t uid;
            (*arc) >> uid;
            typename mpi_state_manager<F>::vertex_type u = 
                                              state_.model_->id2vertex(uid);
            state_.state_lock_.lock();
            // create messages
            state_.messages_[u][v].message = 1;
            typename mpi_state_manager<F>::message_type & m = 
                                        state_.messages_[u][v].message;

            state_.state_lock_.unlock();
            m.load_remap(*arc, state_.id2var_);
            assert(m.arg_vector().size() > 0); 
          }
          state_.deferred_insertions_lock.lock();

          state_.deferred_insertions.push_back(p);
          state_.deferred_insertions_lock.unlock();
          state_.shutdown_cond_.broadcast();
          state_.finishedcounter_cond_.broadcast();
                
          break;
        }
        default: 
          std::cerr << "MPI_STATE_MANAGER_PROTOCOL: Invalid Message!\n";
          break;
      }
      // I must send an ACK
      if (typeofmessage < 0) {
        send_message(MTYPE_ACK, src);
      }
    }
    void terminate() {  
      // some really really big number to force a termination
      finished_ = true;
      ackcond_.signal();
      
    }
    
    /**
      sends a message of a particular type 
      if destination < 0, its a broadcast. the broadcast DOES NOT send to self
    */
    void send_message(int32_t typeofmessage, int destination = -1) {
      std::stringstream strm;
      strm.write(reinterpret_cast<const char*>(&typeofmessage), sizeof(int32_t));
      boost::archive::binary_oarchive arc(strm);
      switch (typeofmessage) {
        // no contents necessary for ACK
        case MTYPE_ACK:
        case MTYPE_FINISHED:
        case MTYPE_RESUMED:
        case MTYPE_SEND_BELIEF:
        case MTYPE_SHUTDOWN:
          break;

        // puts in the parameters
        case MTYPE_PARAMETERS:
          arc << state_.epsilon_;
          arc << state_.max_vertices_;
          break;
          
        // save the factor graph
        case MTYPE_FACTOR_GRAPH:
          arc << *(state_.model_);
          arc << state_.vertex2owner_;
          arc << state_.owner2vertex_;
          arc << state_.var2id_;
          arc << state_.id2var_;
          break;

        default:
          std::cerr << "MPI_STATE_MANAGER_PROTOCOL: send_message: unknown message type!\n";
          assert(false);
      }
      #ifdef DEBUG
      std::cout << "Sending Message Type: " << typeofmessage 
                << " length: " << strm.str().length() << std::endl;
      #endif
      // now we actually send the message
      // if it is a blocking message, I will need to block until
      // the last blocking message completes
      bool blockingmessage = (typeofmessage < 0);
      if (blockingmessage) {
        blocking_messages_.lock();
      }
      
      if (finished_) return; 
      
      // send the message and count the number of receipients
      int numsent;
      if (destination < 0) {
        numsent = po_.num_processes() - 1;
        for (size_t i = 0; i < po_.num_processes(); ++i) {
          if (i != po_.id()) {
            #ifdef DEBUG
            std::cout <<"." << i ;
            #endif
            po_.send_message(i,
                             MPI_STATE_MANAGER_PROT_ID, 
                             strm.str().length(), 
                             strm.str().c_str());
          }
        }
        #ifdef DEBUG
        std::cout << std::endl;
        #endif
      }
      else {
        numsent = 1;
        po_.send_message(destination,
                         MPI_STATE_MANAGER_PROT_ID, 
                         strm.str().length(), 
                         strm.str().c_str());
      }
      
      // if it is a blocking message, wait for the acks
      if (blockingmessage) {
        ackmutex_.lock();
        while (ackcount_ < numsent || finished_) {
            ackcond_.wait(ackmutex_);
        }
        ackcount_ -= numsent;
        ackmutex_.unlock();
        blocking_messages_.unlock();
      }
    }
    
    /**
     sends a BP message (as opposed to send an MPI message)
    */
    void send_message_update(const typename mpi_state_manager<F>::vertex_type &src, 
                             const typename mpi_state_manager<F>::vertex_type &dest, 
                             const typename mpi_state_manager<F>::message_type &m) {
      // get the ids of the vertices and call the alternate version 
      // of this function
      int destination = state_.vertex2owner_[dest]; 
      uint32_t srcid = state_.model_->vertex2id(src);
      uint32_t destid = state_.model_->vertex2id(dest);
      #ifdef DEBUG
      std::cout << "sendupdate " << srcid << " " << destid << "\n";
      #endif
      send_message_update(destination, srcid, destid, m);
    }
    
    /**
     sends a BP message (as opposed to send an MPI message) variation
     that takes vertex ids
    */    
    void send_message_update(int destinationmachine,
                             uint32_t srcvertexid, uint32_t destvertexid, 
                             const typename mpi_state_manager<F>::message_type &m) {
      messages_sent++;
      std::stringstream strm;
      strm.write(reinterpret_cast<const char*>(&MTYPE_BP_MESSAGE), sizeof(int32_t));
      boost::archive::binary_oarchive arc(strm);
      arc << srcvertexid;
      arc << destvertexid;
      m.save_remap(arc, state_.var2id_);
      assert(m.arg_vector().size() > 0);
      po_.send_message(destinationmachine,
                       MPI_STATE_MANAGER_PROT_ID, 
                       strm.str().length(), 
                       strm.str().c_str());
    }
    /**
     sends a BP message (as opposed to send an MPI message) variation
     that takes vertex ids
    */    
    void send_belief(int destinationmachine,
                     const typename mpi_state_manager<F>::vertex_type &v) {
      assert(v.is_variable());
      std::stringstream strm;
      strm.write(reinterpret_cast<const char*>(&MTYPE_BELIEF), sizeof(int32_t));
      boost::archive::binary_oarchive arc(strm);
      uint32_t vid = state_.model_->vertex2id(v);
      arc << vid;
      typename mpi_state_manager<F>::belief_type* b = state_.checkout_belief(v);
      b->save_remap(arc, state_.var2id_);
      assert(b->arg_vector().size() > 0);
      state_.checkin_belief(v,b);
      po_.send_message(destinationmachine,
                  MPI_STATE_MANAGER_PROT_ID, 
                  strm.str().length(), 
                  strm.str().c_str());
    }
    
    /**
      this should only be called if it is possible to guarantee that
      this function has exclusive access to message datastructures
    */
    void give_away_vertex(const typename mpi_state_manager<F>::vertex_type v, 
                          int destination) {
      assert(destination >= 0);
      vertices_transfered++;
      
      // update local owner table
      state_.state_lock_.lock();
      state_.SetOwner(v, destination);
      double residual = state_.schedule_[v];
      state_.schedule_.remove(v);
      state_.state_lock_.unlock();
      /* pack the vertex:
      
         serialization is : 
         vertexid, residual
         if vertex is a variable then also serialize the belief
         for each neighbor:
            neighbor id, neighbor to vertex message
      */
      std::stringstream strm;
      strm.write(reinterpret_cast<const char*>(&MTYPE_VERTEX_TRANSFER), sizeof(int32_t));
      boost::archive::binary_oarchive arc(strm);
      uint32_t vid = state_.model_->vertex2id(v);
      arc << vid;
      // std::cout << "giving away vertex " << vid << std::endl;
      arc << residual;
      if (v.is_variable()) {
        assert(state_.beliefs_[&(v.variable())].arg_vector().size() > 0);
        state_.beliefs_[&(v.variable())].save_remap(arc, state_.var2id_);
        
      }

      foreach (const typename mpi_state_manager<F>::vertex_type &u, 
               state_.model_->neighbors(v)) {
        uint32_t uid = state_.model_->vertex2id(u);
        arc << uid;
        // lets do direct reads instead of through checkout
        // because we are going to delete them also
        state_.state_lock_.lock();
        typename mpi_state_manager<F>::message_type &msg = 
                                              state_.messages_[u][v].message;
        
        assert(msg.arg_vector().size() > 0);
        msg.save_remap(arc, state_.var2id_);
        state_.messages_[u].erase(v);
        if (state_.messages_[u].size() == 0) {
          state_.messages_.erase(u);
        }
        state_.state_lock_.unlock();
      }
      if (po_.id() !=0 && v.is_variable()) state_.beliefs_.erase(&(v.variable()));
      po_.send_message(destination,
                  MPI_STATE_MANAGER_PROT_ID, 
                  strm.str().length(), 
                  strm.str().c_str());
      // done.
      
    }
    
    void PrintCounts() {
      std::cout << "Vertices sent: " << vertices_transfered << std::endl;
      std::cout << "Vertices received: " << vertices_received << std::endl;
      std::cout << "Messages sent: " << messages_sent << std::endl;
    }
  };

}

};
#include <sill/macros_undef.hpp>
#endif

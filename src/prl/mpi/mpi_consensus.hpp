#ifndef MPI_CONSENSUS_HPP
#define MPI_CONSENSUS_HPP

#include <sstream>
#include <prl/parallel/pthread_tools.hpp>
#include <prl/mpi/mpi_protocols.hpp>
#include <prl/mpi/mpi_wrapper.hpp>
#include <prl/serialization/serialize.hpp>

namespace prl {
  class mpi_consensus;

  struct consensus_state {
      bool globaldone;
      size_t sent;
      size_t recv;
  };


 class consensus_po_box_callback: public mpi_post_office::po_box_callback {
    private:
      mpi_consensus &consensus;
      mpi_post_office &po;
    public:
      consensus_po_box_callback(mpi_consensus &cons, mpi_post_office &p):
        consensus(cons), po(p){ }

      void recv_message(const mpi_post_office::message& msg);
      
      ~consensus_po_box_callback() { }
  };

   /**
   * Distributed Consensus Protocol
   * Works by passing in a ring, the global number of sent and received messages
   * Once the remote state agrees with my private state, and #sent == #received
   * and I am done, then we must have reached global consensus
  */
  class mpi_consensus {
    private:
      mpi_post_office &po;
      mutex statelock;      /// Locks the state of the next 6 variables
      consensus_state state;  /// My current known state of the system
      consensus_state lastreceivedstate;

      size_t lasttransmitted_remotesent;
      size_t lasttransmitted_remoterecv;
      size_t lasttransmitted_localsent;
      size_t lasttransmitted_localrecv;
      size_t localsent;     /// number of messages I sent
      size_t localrecv;     /// number of messages I received
      volatile bool localdone;       /// whether we are done locally
      volatile bool hastoken;        /// whether I current have the token
      friend class consensus_po_box_callback;
      conditional doneconditional;
      
      size_t numrecv_on_entercs;
    public:

      mpi_consensus(mpi_post_office &poffice);

      ~mpi_consensus();
      

      size_t global_num_sent();
      size_t global_num_recv();

      // this version of finished reads the numsent and received from the mpi
      // wrapper
      bool finished();
      // if you keep track of the number sent and received, use
      // this version of finished;
      bool finished(size_t totalsent, size_t totalrecv);

      bool has_token() {
        return hastoken;
      } 
      // called from another thread. This forces a thread waiting on finish
      // to wakeup (and probably return false); Wakup should not be called
      // while in a critical section
      void wakeup();


      void begin_critical_section();
      void end_critical_section();
      bool end_critical_section_and_finish(bool sleep = true);
      bool end_critical_section_and_finish(size_t totalsent, size_t totalrecv, bool sleep = true);
  }; // end class mpi_simple_consensus


} // end namespace prl



#endif

#include <sstream>
#include <sill/parallel/pthread_tools.hpp>
#include <sill/mpi/mpi_protocols.hpp>
#include <sill/mpi/mpi_wrapper.hpp>
#include <sill/serialization/serialize.hpp>
#include <sill/mpi/mpi_consensus.hpp>

namespace sill {

  void consensus_po_box_callback::recv_message(const mpi_post_office::message& msg) {
    // read and deserialize the the token
    if (msg.body_size == 0) return;
    std::stringstream istr(std::string(msg.body, msg.body_size));
    iarchive iarc(istr);
    consensus_state m;
    iarc >> m.globaldone >> m.sent >> m.recv;

    // if the globaldone flag is set, set it locally and forward
    if (m.globaldone) {
        // if I am already done. break
        consensus.statelock.lock();
        consensus.state.globaldone = true;
        consensus.doneconditional.signal();
        consensus.statelock.unlock();
        return;
    }
    else {
      // its not globally done. Update my local known state
      // and if I am done, forward the token
      consensus.statelock.lock();
      // calculate how the remote sent and recv
      size_t remotedeltasent = m.sent - consensus.lasttransmitted_localsent
                                      - consensus.lasttransmitted_remotesent;
      size_t remotedeltarecv = m.recv - consensus.lasttransmitted_localrecv
                                      - consensus.lasttransmitted_remoterecv;
      consensus.state.sent += remotedeltasent;
      consensus.state.recv += remotedeltarecv;

      // if I am locally done, I can forward the token
      if (consensus.localdone) {
        // check if we are globally done too
        // sent == recv, and state == lastreceived state
        if (consensus.state.sent == consensus.state.recv &&
            consensus.lastreceivedstate.sent == consensus.state.sent &&
            consensus.lastreceivedstate.recv == consensus.state.recv) {
          consensus.state.globaldone = true;
          consensus.doneconditional.signal();
        }
        // forward the token
        std::stringstream ostr;
        oarchive oarc(ostr);
        oarc << consensus.state.globaldone
              << consensus.state.sent
              << consensus.state.recv;
        ostr.flush();
        // update the last transmitted values.
        consensus.lasttransmitted_localrecv = consensus.localrecv;
        consensus.lasttransmitted_localsent = consensus.localsent;
        consensus.lasttransmitted_remoterecv = consensus.state.recv - consensus.localrecv;
        consensus.lasttransmitted_remotesent = consensus.state.sent - consensus.localsent;

        if (consensus.state.globaldone) {
          // if we are done, we broadcast
          po.bcast_message(MPI_PROTOCOL::CONSENSUS,
                            ostr.str().length(),
                            ostr.str().c_str(),
                            mpi_post_office::ALL_BUT_SELF);
        }
        else {
          // forward to the next machine in the sequence
          po.send_message((po.id() + 1) % po.num_processes(),
                            MPI_PROTOCOL::CONSENSUS,
                            ostr.str().length(),
                            ostr.str().c_str());
        }
      }
      else {
        // we are not done here. hold on to the token
        consensus.hastoken = true;
      }
      consensus.lastreceivedstate = m;
      consensus.statelock.unlock();
    }
  }



  mpi_consensus::mpi_consensus(mpi_post_office &poffice) :
    po(poffice), localsent(0), localrecv(0), localdone(false) {
    if (po.id() == 0) {
      hastoken = true;
    }
    else {
      hastoken = false;
    }
    state.sent = 0;
    state.recv = 0;
    lasttransmitted_localsent = 0;
    lasttransmitted_localrecv = 0;
    lasttransmitted_remotesent = 0;
    lasttransmitted_remoterecv = 0;

    state.globaldone = false;
    lastreceivedstate = state;
    statelock.lock();
    po.register_handler(MPI_PROTOCOL::CONSENSUS,
                        new consensus_po_box_callback(*this, po));
    po.register_condvar(-1, &doneconditional);
    statelock.unlock();
    poffice = po;
  }

  mpi_consensus::~mpi_consensus() {
    statelock.lock();
    po.unregister_condvar(&doneconditional);
    po.unregister_handler(MPI_PROTOCOL::CONSENSUS);
    statelock.unlock();
  }

  bool mpi_consensus::finished() {
    return finished(po.num_msg_sent(), po.num_msg_recv());
  }

  size_t mpi_consensus::global_num_sent() {
    return state.sent;
  }
  size_t mpi_consensus::global_num_recv() {
    return state.recv;
  }
  // if you keep track of the number sent and received, use
  // this version of finished;
  bool mpi_consensus::finished(size_t totalsent, size_t totalrecv) {
      if (state.globaldone) return true;
      begin_critical_section();
      return end_critical_section_and_finish(totalsent, totalrecv);
  }

  void mpi_consensus::wakeup() {
    statelock.lock();
    doneconditional.signal();
    statelock.unlock();
  }

  void mpi_consensus::begin_critical_section() {
    statelock.lock();
    numrecv_on_entercs = po.num_msg_recv();
  }
  void mpi_consensus::end_critical_section() {
    statelock.unlock();
  }

  bool mpi_consensus::end_critical_section_and_finish(bool sleep) {
    // update my localstate
    size_t totalsent = po.num_msg_sent();
    size_t totalrecv = po.num_msg_recv();
    // if I have received new packets between the critical sections, 
    // it is not safe to continue
    if (totalrecv != numrecv_on_entercs) {
      statelock.unlock();
      return false;
    }
    state.sent += totalsent - localsent;
    state.recv += totalrecv - localrecv;
    localsent = totalsent;
    localrecv = totalrecv;
    localdone = true;

    // if I have the token, forward it
    if (hastoken) {
      // check if we are globally done too
      // sent == recv, and state == lastreceived state
      if (state.sent == state.recv &&
          lastreceivedstate.sent == state.sent &&
          lastreceivedstate.recv == state.recv) {
        state.globaldone = true;
      }
      // forward the token
      std::stringstream ostr;
      oarchive oarc(ostr);

      oarc << state.globaldone << state.sent << state.recv;
      ostr.flush();
      // update the last transmitted values.
      lasttransmitted_localrecv = localrecv;
      lasttransmitted_localsent = localsent;
      lasttransmitted_remoterecv = state.recv - localrecv;
      lasttransmitted_remotesent = state.sent - localsent;
      if (state.globaldone) {
        // if we are done, we broadcast termination
        po.bcast_message(MPI_PROTOCOL::CONSENSUS,
                          ostr.str().length(),
                          ostr.str().c_str(),
                          mpi_post_office::ALL_BUT_SELF);
      }
      else {
        // forward to the next machine in the sequence
        po.send_message((po.id() + 1) % po.num_processes(),
                          MPI_PROTOCOL::CONSENSUS,
                          ostr.str().length(),
                          ostr.str().c_str());
      }
      hastoken = false;
    }
    statelock.unlock();
    // if I am done, quit
    if (state.globaldone) {
      return true;
    }
    else {
      // if any message comes in now, the conditional variable should wake up
      // but this is a fast check so I don't have to go to sleep
      if (po.num_msg_recv() != numrecv_on_entercs) return false;
      if (sleep) {
        // go to sleep. Wake up if I receive anything
        doneconditional.wait(statelock);
        bool ret = state.globaldone;
        statelock.unlock();

        return ret;
      }
      else {
        return state.globaldone;
      }
    }

  }
  
  bool mpi_consensus::end_critical_section_and_finish(size_t totalsent,
                                                      size_t totalrecv,
                                                      bool sleep) {
    // update my localstate
    state.sent += totalsent - localsent;
    state.recv += totalrecv - localrecv;
    localsent = totalsent;
    localrecv = totalrecv;
    localdone = true;

    // if I have the token, forward it
    if (hastoken) {
      // check if we are globally done too
      // sent == recv, and state == lastreceived state
      if (state.sent == state.recv &&
          lastreceivedstate.sent == state.sent &&
          lastreceivedstate.recv == state.recv) {
        state.globaldone = true;
      }
      // forward the token
      std::stringstream ostr;
      oarchive oarc(ostr);

      oarc << state.globaldone << state.sent << state.recv;
      ostr.flush();
      // update the last transmitted values.
      lasttransmitted_localrecv = localrecv;
      lasttransmitted_localsent = localsent;
      lasttransmitted_remoterecv = state.recv - localrecv;
      lasttransmitted_remotesent = state.sent - localsent;
      if (state.globaldone) {
        // if we are done, we broadcast termination
        po.bcast_message(MPI_PROTOCOL::CONSENSUS,
                          ostr.str().length(),
                          ostr.str().c_str(),
                          mpi_post_office::ALL_BUT_SELF);
      }
      else {
        // forward to the next machine in the sequence
        po.send_message((po.id() + 1) % po.num_processes(),
                          MPI_PROTOCOL::CONSENSUS,
                          ostr.str().length(),
                          ostr.str().c_str());
      }
      hastoken = false;
    }
    statelock.unlock();
    // if I am done, quit
    if (state.globaldone) {
      return true;
    }
    else {
      if (sleep) {
        // go to sleep. Wake up if I receive anything
        doneconditional.wait(statelock);
        bool ret = state.globaldone;
        statelock.unlock();

        return ret;
      }
      else {
        return state.globaldone;
      }
    }
  }

} // end namespace sill


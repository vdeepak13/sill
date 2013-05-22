// Probabilistic Reasoning Library (PRL)
// Copyright 2005, 2009 (see AUTHORS.txt for a list of contributors)
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA


#include <mpi.h>


#include <iostream>
#include <string>
#include <cassert>
#include <cstring>

#include <sill/base/stl_util.hpp>
#include <sill/mpi/mpi_wrapper.hpp>
#include <sill/mpi/mpi_protocols.hpp>

#include <sill/macros_def.hpp>
namespace sill {


  int build_mpi_tag(mpi_post_office::message &msg) {
    return msg.type | 
       (MPI_WRAPPER_PROTOCOL_FLAGS::FLAG_UNCOUNTED & ((int)(!msg.uncounted)-1));
  }
  void reverse_mpi_tag(mpi_post_office::message &msg, int tag) {
    msg.type = tag & MPI_WRAPPER_PROTOCOL_FLAGS::PROTOCOL_BITMASK;
    msg.uncounted = !!(tag & MPI_WRAPPER_PROTOCOL_FLAGS::FLAG_UNCOUNTED);
  }

  /////////////////////////////////////////////////////////////////////////////
  //  Helper Routines
  /////////////////////////////////////////////////////////////////////////////
  // Send message to addres given by dest assigning request to request.
  // This will block until completion
  void  do_send(mpi_post_office::message& msg, 
                MPI_Request &request) {

    int ret = MPI_Isend(reinterpret_cast<void*>(const_cast<char*>(msg.body)),
                  msg.body_size,
                  MPI_CHAR,
                  msg.dest,
                  build_mpi_tag(msg),
                  MPI_COMM_WORLD,
                  &request);

    if (ret != MPI_SUCCESS) {
      std::cout << "do_send Isend failure with code " << ret << std::endl;
    }
    
    // Wait for the request to finish
    ret = MPI_Wait(&request, MPI_STATUS_IGNORE);
    if (ret != MPI_SUCCESS) {
      std::cout << "do_send Wait failure with code " << ret << std::endl;
    } 
  } 

  /////////////////////////////////////////////////////////////////////////////
  //  Sender Thread Class
  /////////////////////////////////////////////////////////////////////////////

  // constructor
  mpi_post_office::sender_thread::  
  sender_thread() : m_alive(true) {  }
  

  void mpi_post_office::sender_thread::
  set_numprocs(size_t numprocs) { m_numprocs = numprocs; }

  void mpi_post_office::sender_thread::
  set_id(size_t id) { m_id = id; }
  

  // add a message to the sending queue
  void mpi_post_office::sender_thread::
  add(const mpi_post_office::message& msg) {
    // Enqueue the new message
    m_queue.enqueue(msg.duplicate());
  }

  // Run
  void mpi_post_office::sender_thread::
  run() {
    // while the post office is still alive or the queue is not empty
    while(m_alive || !m_queue.empty()) {
      // This is a blocking call to dequeue
      message msg;
      bool success;
      std::pair<message, bool> pair = m_queue.dequeue();
      msg = pair.first;
      success = pair.second;
      // If success = true than the dequeu was successful
      if(success) {
        // Determine if this is a broadcast send
        if(msg.is_bcast) {
          for(size_t i = 0; i < m_numprocs; ++i) {
            // if we don't need to skip self message then send
            if( !(msg.bcast == ALL_BUT_SELF && m_id == i) ) {
              msg.dest = i;
              do_send(msg, m_request);
            }
          } // End of for loop over processes to send to
        } else {
          // not a bcast
          do_send(msg, m_request);
        } 
        // Free the memory associated with the message
        if(msg.body_size > 0 && msg.body != NULL) {
          delete [] msg.body;
          msg.body = NULL;
        } 
      }
    } 
    std::cout << "Sender exit" << std::endl;
  } 
    
  void mpi_post_office::sender_thread::flush() {
    m_queue.wait_until_empty();
  }
  
  void mpi_post_office::sender_thread::finish() {
    // set alive to false
    m_alive = false;
    // Wake up the sender thread from both the 
    // queue or the send
    m_queue.stop_blocking();
  }


  // destructor
  mpi_post_office::sender_thread::~sender_thread() {
    m_alive = false;
  }


  /////////////////////////////////////////////////////////////////////////////
  //  Receiver Thread 
  /////////////////////////////////////////////////////////////////////////////

  // Constructor
  mpi_post_office::receiver_thread::
  receiver_thread() { }

  // Initialize the receiver_thread
  void mpi_post_office::receiver_thread::
  initialize(size_t id, 
             sender_thread* sender,
             std::map<int, po_box_callback*>* handlers,
              mpi_post_office *po,
             size_t buffer_size = (200 * 1048576) ) {
    m_id = id;
    m_po = po;
    m_handlers = handlers;
    assert(m_handlers != NULL);
    m_sender = sender;
    assert(m_sender != NULL);
    // Allocate for the local message buffer
    m_body_size = buffer_size;
    m_body = new char[m_body_size];
    assert(m_body != NULL);

    m_alive = true;
    m_receiving = false;

  }


  // Handler
  void mpi_post_office::receiver_thread::run() {
    m_alive = true;
    m_receiving = false;
    while(1) {
      MPI_Status status;
      // Initialize must be called first
      assert(m_body != NULL);
      m_statuslock.lock();
      
      if (!m_alive) {
        m_statuslock.unlock();
        break;
      }
      m_receiving = true;
      m_statuslock.unlock();
      int flag = 0;
      int ret = 0;
      while(m_alive && !flag) { 
        ret = MPI_Iprobe(MPI_ANY_SOURCE, 
                         MPI_ANY_TAG, 
                         MPI_COMM_WORLD, &flag, &status);
        if (ret != MPI_SUCCESS) {
          std::cout << "receiver thread fail at Iprobe with code " << ret;
        }
        if (!flag) {
          usleep(200);
        }
      }
      if (flag) { // if I received something
        ret = MPI_Recv(m_body, m_body_size, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      }
      if (ret != MPI_SUCCESS) {
        std::cout << "receiver thread fail at Wait with code " << ret
                                                               << std::endl;
      }
      m_statuslock.lock();
      m_receiving = false;
      m_statuslock.unlock();
      if (flag) { // if I received something
        // Create a temporary message and fill in the values
        message msg;
        msg.orig = status.MPI_SOURCE;
        msg.dest = m_id;
        reverse_mpi_tag(msg, status.MPI_TAG);
        
        if( msg.type == MPI_PROTOCOL::STOP) {
          // begin terminting this post office by forcing the sender
          // thread to finish.
          m_sender->finish();
        } else {
          int count;
          MPI_Get_count(&status, MPI_CHAR, &count);
          assert(count >= 0);
          msg.body_size = count;
          msg.body = m_body;

          // Trigger the handlers
          typedef std::map<int, po_box_callback*>::iterator iterator;
          assert(m_handlers != NULL);
          if (msg.type > MPI_PROTOCOL::INTERNAL_UPPER_ID) {
            m_po->numrecv+=(msg.uncounted == false);
          }
          else {
            m_po->ctrlrecv++;
          }
          m_po->handler_lock.lock();
          iterator i = m_handlers->find(msg.type);
          if(i != m_handlers->end()) {
            if (i->second == NULL) {
              m_po->m_syncreceives[msg.type]->enqueue(msg.duplicate());
            }
            else {
             i->second->recv_message(msg);
            }
          } else {
            std::cout << "No handler registered, message dropped!" << std::endl;
          } // End of a handler is registered

          // execute the condvar handlers
          // trigger for identically equal messages
          typedef std::multimap<int, conditional*>::iterator mmiterator_type;
          std::pair<mmiterator_type, mmiterator_type> range =
                            m_po->m_condvar_handlers.equal_range(msg.type);
          for(mmiterator_type i = range.first; i != range.second; ++i) {
            i->second->signal();
          }
          // trigger for identically global messages
          range = m_po->m_condvar_handlers.equal_range(-1);
          for(mmiterator_type i = range.first; i != range.second; ++i) {
            i->second->signal();
          }
          m_po->handler_lock.unlock();
        } // End of if stop message
      }
    } // End of while loop
    std::cout << "Receiver exit" << std::endl;
  }

  void mpi_post_office::receiver_thread::
  stop() {
    // set alive to false
    m_statuslock.lock();
    m_alive = false;
    m_statuslock.unlock();
  }


  // destructor
  mpi_post_office::receiver_thread::~receiver_thread() {
    // deallocate message body
    if(m_body != NULL) delete [] m_body;
    m_body = NULL;
  }


  /////////////////////////////////////////////////////////////////////////////
  //  mpi_post_office class
  /////////////////////////////////////////////////////////////////////////////


  mpi_post_office::mpi_post_office() {
    // Initialize the MPI environment
    int provided;
    MPI_Init_thread( 0, 0, MPI_THREAD_MULTIPLE, &provided);
    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    // get the number of processes
    int numprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    
    assert(numprocs >= 1);
    m_numprocs = numprocs;
    // Get the id
    int id;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    assert(id >= 0);
    m_id = id;

// Initialize the receiver
    m_receiver.initialize(m_id, &m_sender, &m_handlers, this);
    m_sender.set_numprocs(m_numprocs);
    m_sender.set_id(m_id);
    numsent = 0;
    numrecv = 0;
    ctrlsent = 0;
    ctrlrecv = 0;
  }

  size_t mpi_post_office::id() {
    return m_id;
  }

  size_t mpi_post_office::num_msg_sent() {
    return numsent;
  }
  
  size_t mpi_post_office::num_msg_recv() {
    return numrecv;
  }

  size_t mpi_post_office::num_ctrl_sent() {
    return ctrlsent;
  }

  size_t mpi_post_office::num_ctrl_recv() {
    return ctrlrecv;
  }

  size_t mpi_post_office::
  num_processes() {
    return m_numprocs;
  }

  void mpi_post_office::
  register_handler(int message_type,
                   mpi_post_office::po_box_callback* callback) {
    assert(message_type >= 0);

    std::cerr << "register " << message_type << "\n";
    std::cerr.flush();
    assert(m_handlers.find(message_type) == m_handlers.end());
    unregister_handler(message_type);
    handler_lock.lock();
    // this is a synchronous message
    // create the queue
    if (callback == NULL) {
      m_syncreceives[message_type] = new blocking_queue<message>;
    }
    m_handlers[message_type] = callback;
    handler_lock.unlock();
  }

  void mpi_post_office::
  unregister_handler(int message_type) {
    assert(message_type >= 0);
    handler_lock.lock();
    if(m_handlers.find(message_type) != m_handlers.end()) {
      std::cerr << "unregister " << message_type << "\n";
      std::cerr.flush();
      if (m_handlers[message_type] == NULL) {
        delete m_syncreceives[message_type];
        m_syncreceives.erase(message_type);
      }
      else {
        m_handlers[message_type]->terminate();
      }
      m_handlers.erase(message_type);
    }
    handler_lock.unlock();
  }

  void mpi_post_office::
  register_condvar(int message_type,
                   conditional* cond) {
    m_condvar_handlers.insert(std::pair<int, conditional*>(message_type, cond));
  }

  void mpi_post_office::
  unregister_condvar(conditional* cond) {
    std::multimap<int, conditional*>::iterator i = m_condvar_handlers.begin();
    while (i!=m_condvar_handlers.end()) {
      if (i->second == cond) {
        m_condvar_handlers.erase(i);
        break;
      }
      ++i;
    }
  }

  bool mpi_post_office::receive(size_t type, message &msg) {
    assert(m_syncreceives.find(type) != m_syncreceives.end());
    std::pair<message, bool> p = m_syncreceives[type]->dequeue();
    msg = p.first;
    return p.second;
  }
  
  bool mpi_post_office::async_receive(size_t type, message &msg) {
    assert(m_syncreceives.find(type) != m_syncreceives.end());
    std::pair<message, bool> p = m_syncreceives[type]->try_dequeue();
    msg = p.first;
    return p.second;
  }
  
  void mpi_post_office::
  start() {
    // Luanch the receiver thread
    m_receiver.start(15);
    // Launch the sender thread
    m_sender.start(15);
  }
  

  void mpi_post_office::
  stopAll() {
    std::cout << "Stop Invoked!!!!!!!!!!!!" << std::endl;
    // stop the sender thread
    // Do a manual bcast to avoid a race with send and receive
    message msg;
    msg.orig = m_id;
    msg.dest = 0;
    msg.bcast = ALL;
    msg.is_bcast = true;
    msg.type = MPI_PROTOCOL::STOP;
    msg.body_size = 0;
    msg.body = NULL;
    for(size_t i = 0; i < m_numprocs; ++i) {
        msg.dest = i;
        MPI_Send(reinterpret_cast<void*>(const_cast<char*>(msg.body)),
                 msg.body_size,
                 MPI_CHAR,
                 msg.dest,
                 msg.type,
                 MPI_COMM_WORLD);
    }
  }


  void mpi_post_office::
  send_message(size_t dest, int type, 
               size_t body_size, 
               const char* body, bool uncounted) {
    // Create a message object and fill in the necessary fields 
    message msg;
    msg.orig = m_id;
    msg.dest = dest;
    msg.is_bcast = false;
    msg.type = type;
    msg.body_size = body_size;
    msg.body = body;
    msg.uncounted = uncounted;
    // Ensure that message is valid to send
    assert(msg.dest < m_numprocs);
    // Add the message to the sender queue note that the message
    // memory will be duplicated at this point
    if (msg.type > MPI_PROTOCOL::INTERNAL_UPPER_ID) {
      // increment if the message's uncounted flag is not set
      numsent += (msg.uncounted == false);
    }
    else {
      ctrlsent++;
    }
    m_sender.add(msg);
  }

  void mpi_post_office::flush() {
    m_sender.flush();
  }
  
  void mpi_post_office::
  bcast_message(int type, size_t body_size, const char* body, 
                bcast_type bcast) {
      // Create a message object and fill in the necessary fields 
      message msg;
      msg.orig = m_id;
      msg.dest = 0;
      msg.bcast = bcast;
      msg.is_bcast = true;
      msg.type = type;
      msg.body_size = body_size;
      msg.body = body;
      // Add the message to the sender queue note that the message
      // memory will be duplicated at this point
      if (msg.type > MPI_PROTOCOL::INTERNAL_UPPER_ID) {
        if (msg.uncounted == false) {
          numsent += num_processes();
        }
      }
      else {
        ctrlsent += num_processes();
      }
      m_sender.add(msg);
  }


  void mpi_post_office::wait() {
    // First wait for all sends to terminate
    m_sender.join();
    // We can't turn off the receiver until everyone has turned off
    // their sender (no active sends waiting to take place)
    barrier();
    // Shutdown the receiver
    m_receiver.stop();
    m_receiver.join();
    // Finished waiting for the mpi_post_office process
  }

  void mpi_post_office::barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
  }



  // wait for all threads to terminate before closing
  mpi_post_office::
  ~mpi_post_office() {
    // Shut down the mpi connection
    foreach(int i, keys(m_handlers)) {
      unregister_handler(i);
    }

    MPI_Finalize();
    std::cout << "SHUTDOWN!" << std::endl;
  }



  std::ostream&
  operator<<(std::ostream& out, const mpi_post_office::message& msg) {
    return out <<"[" 
               << msg.orig << "->" << msg.dest 
               << "(" << msg.type << ") "
               << msg.body_size << " : " << msg.body 
               << "]";
  }

  void mpi_inplace_reduce_uint64(uint64_t &x) {
    // on 32-bit architectures, MPI may not define SUM for long longs
    if (sizeof(long) == 4) {
      uint32_t x2 = x;
      mpi_inplace_reduce_uint32(x2);
      x=x2;
    }
    else {
      uint64_t result = 0;
      int ret = MPI_Reduce(&x, &result, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
      if (ret != 0) {
      int eclass,len;
      char estring[MPI_MAX_ERROR_STRING];
  
      MPI_Error_class(ret, &eclass);
      MPI_Error_string(ret, estring, &len);
      printf("Error %d: %s\n", eclass, estring);fflush(stdout);
      }
      x = result;
    }
  }
  void mpi_inplace_reduce_uint32(uint32_t &x) {
    uint64_t result = 0;
    int ret = MPI_Reduce(&x, &result, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (ret != 0) {
    int eclass,len;
    char estring[MPI_MAX_ERROR_STRING];

    MPI_Error_class(ret, &eclass);
    MPI_Error_string(ret, estring, &len);
    printf("Error %d: %s\n", eclass, estring);fflush(stdout);
    }
    x = result;
  }
}
#include <sill/macros_undef.hpp>

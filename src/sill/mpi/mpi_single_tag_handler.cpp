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

#include <sill/mpi/mpi_single_tag_handler.hpp>


#include <sill/macros_def.hpp>
namespace sill {


  /////////////////////////////////////////////////////////////////////////////
  //  Helper Routines
  /////////////////////////////////////////////////////////////////////////////
  mpi_single_tag_handler::message 
  duplicateAddHeader(const mpi_single_tag_handler::message& msg) {
    // Make a copy of the original message
    mpi_single_tag_handler::message new_msg = msg;
    // Make a copy of the buffer
    char* body = new  char[msg.body_size+sizeof(int32_t)];
    assert(body != NULL);
    new_msg.body_size = msg.body_size+sizeof(int32_t);
    // do a memory copy
    memset(body,0,sizeof(int32_t));
    if (msg.quit) body[0] = 1;
    std::memcpy(body+sizeof(int32_t), msg.body, msg.body_size);
    // Make the new message body the new body;
    new_msg.body = body;
    // Return a new message which is a duplicate the original 
    return new_msg;
  }
/////////////////////////////////////////////////////////////////////////////
  //  Helper Routines
  /////////////////////////////////////////////////////////////////////////////
  mpi_single_tag_handler::message 
  duplicate(const mpi_single_tag_handler::message& msg) {
    // Make a copy of the original message
    mpi_single_tag_handler::message new_msg = msg;
    // Make a copy of the buffer
    char* body = new  char[msg.body_size];
    assert(body != NULL);
    new_msg.body_size = msg.body_size;
    // do a memory copy
    std::memcpy(body, msg.body, msg.body_size);
    // Make the new message body the new body;
    new_msg.body = body;
    // Return a new message which is a duplicate the original 
    return new_msg;
  }
  // Send message to addres given by dest assigning request to request.  This will block until 
  // completion
  void  do_send(mpi_single_tag_handler::message& msg, 
                MPI::Request& request) {
    try {
      request = MPI::COMM_WORLD.Isend(msg.body, 
                                      msg.body_size,
                                      MPI::CHAR,
                                      msg.dest,
                                      msg.type);
    } catch (MPI::Exception e) {
      std::cout << "Exception 1: " << e.Get_error_string() << std::endl;
            } // End of try catch
    // Wait for the request to finish
    try {
      request.Wait();
      // while(!m_request.Test());
    } catch (MPI::Exception e) {
      std::cout << "Exception 2: " << e.Get_error_string() << std::endl;
    } // End of Try Catch
  } // end of do_send

  /////////////////////////////////////////////////////////////////////////////
  //  Sender Thread Class
  /////////////////////////////////////////////////////////////////////////////

  // constructor
  mpi_single_tag_handler::sender_thread::
  sender_thread() : m_alive(true) {  }
  

  void mpi_single_tag_handler::sender_thread::
  set_numprocs(size_t numprocs) { m_numprocs = numprocs; }

  void mpi_single_tag_handler::sender_thread::
  set_id(size_t id) { m_id = id; }
  

  // add a message to the sending queue
  void mpi_single_tag_handler::sender_thread::
  add(const mpi_single_tag_handler::message& msg) {
    // Enqueue the new message
    m_queue.enqueue(duplicateAddHeader(msg));
  }

  // Run
  void mpi_single_tag_handler::sender_thread::
  run() {
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
          do_send(msg, m_request);
        } // end of if 
        // Free the memory associated with the message
        if(msg.body_size > 0 && msg.body != NULL) {
          delete [] msg.body;
          msg.body = NULL;
        } // End of if success
      }
    } // End of while (alive or the queue is not empty)
    std::cout << "Sender exit" << std::endl;
  } // End of Run
    

  void mpi_single_tag_handler::sender_thread::
  finish() {
    // set alive to false
    m_alive = false;
    // Wake up the sender thread from both the 
    // queue or the send
    m_queue.stop_blocking();
  }


  // destructor
  mpi_single_tag_handler::sender_thread::~sender_thread() {
    m_alive = false;
  }

  /////////////////////////////////////////////////////////////////////////////
  //  Handler Thread Class
  /////////////////////////////////////////////////////////////////////////////
  
  // Constructor
  mpi_single_tag_handler::handler_thread::
  handler_thread() { }
  
  void mpi_single_tag_handler::handler_thread::
  set_callback(mpi_single_tag_handler::po_box_callback* callback) {
    m_callback = callback;
  }

  // Receive a message to handle
  void mpi_single_tag_handler::handler_thread::
  add(const mpi_single_tag_handler::message& msg) {
    // place a copy of the new message in the local queue
    m_queue.enqueue(duplicate(msg));
  } // end of handle

  // Run method
  void mpi_single_tag_handler::handler_thread::
  run() {
    m_alive = true;
    // Ensure that a callback function has been registered
    assert(m_callback != NULL);
    // While this thread is alive
    while(m_alive || !m_queue.empty()) {
      message msg;
      bool success;
      std::pair<message, bool> pair = m_queue.dequeue();
      msg = pair.first;
      success = pair.second;

      if(success) {
        // Handle the actual message (this could take a while)
        msg.body = msg.body+sizeof(int32_t);
        msg.body_size = msg.body_size - sizeof(int32_t);
        m_callback->recv_message(msg);
        msg.body = msg.body-sizeof(int32_t);
        msg.body_size = msg.body_size + sizeof(int32_t);
        // Free the memory associated witht he message
        if(msg.body_size > 0 && msg.body != NULL) {
          delete [] msg.body;
          msg.body = NULL;
        } 
      } // End of if success
    } // End of while alive and queue is not empty
    m_callback->terminate();
    std::cout << "handler_thread exit" << std::endl;
  } // end of run

  void mpi_single_tag_handler::handler_thread::
  stop() {
    // stop this thread
    m_alive = false;
    m_queue.stop_blocking();
  }

  // Destructor
  mpi_single_tag_handler::handler_thread::
  ~handler_thread() {
    m_alive = false;
    assert(m_queue.empty());
  } // end of destructor

  /////////////////////////////////////////////////////////////////////////////
  //  Receiver Thread 
  /////////////////////////////////////////////////////////////////////////////

  // Constructor
  mpi_single_tag_handler::receiver_thread::
  receiver_thread() { }

  // Initialize the receiver_thread
  void mpi_single_tag_handler::receiver_thread::
  initialize(size_t id, 
             sender_thread* sender,
             handler_thread* handler, 
            int tagid,
             size_t buffer_size = (100 * 1048576) ) {
    m_id = id;
    m_tagid = tagid;
    m_handler = handler;
    assert(m_handler != NULL);
    m_sender = sender;
    assert(m_sender != NULL);
    // Allocate for the local message buffer
    m_body_size = buffer_size;
    m_body = new char[m_body_size];
    assert(m_body != NULL);
  }


  // Handler
  void mpi_single_tag_handler::receiver_thread::run() {
    m_alive = true;
    while(m_alive) {
      MPI::Status status;
      // Initialize must be called first
      assert(m_body != NULL);
      // Receive the message
      try { 
        m_request = MPI::COMM_WORLD.Irecv(m_body,
                                          m_body_size,
                                          MPI::CHAR,
                                          MPI::ANY_SOURCE,
                                          m_tagid);
      } catch (MPI::Exception e) {
        std::cout << "Exception 3: " << std::endl;
      }
      
      // Wait for request to return
      try {
        // while(!m_request.Test(status) && m_alive);
        m_request.Wait(status);
      } catch (MPI::Exception e) {
        std::cout << "Exception 4: " << e.Get_error_string() << std::endl;
      }

      // Verify that the message was not cancelled before receiving
      // Probably should just terminat this thread if the message was
      // cancelled
      if(!status.Is_cancelled()) {
        // Create a temporary message and fill in the values
        message msg;
        msg.orig = status.Get_source();
        msg.dest = m_id;
        msg.type = status.Get_tag();
        
        if( m_body[0] == 1) {
          // begin terminting this post office by forcing the sender
          // thread to finish.
          m_sender->finish();
        } else {
          // Should this be stats.Get_elements;
          msg.body_size = status.Get_count(MPI::CHAR);
          msg.body = m_body;
          // Identify the correct handler
          m_handler->add(msg);
        } // End of if stop message
      } // End of if cancelled
    } // End of while loop
    std::cout << "Receiver exit" << std::endl;
  }

  void mpi_single_tag_handler::receiver_thread::
  stop() {
    // set alive to false
    m_alive = false;
    if(m_request != MPI::Request()) m_request.Cancel();
  }


  // destructor
  mpi_single_tag_handler::receiver_thread::~receiver_thread() {
    // deallocate message body
    if(m_body != NULL) delete [] m_body;
    m_body = NULL;
  }


  /////////////////////////////////////////////////////////////////////////////
  //  mpi_single_tag_handler class
  /////////////////////////////////////////////////////////////////////////////


  mpi_single_tag_handler::mpi_single_tag_handler(int type) {
    tagid = type;
    // get the number of processes
    int numprocs = MPI::COMM_WORLD.Get_size();
    assert(numprocs >= 1);
    m_numprocs = numprocs;
    // Get the id
    int id = MPI::COMM_WORLD.Get_rank();
    assert(id >= 0);
    m_id = id;
    // Get the processor name
    char name_buffer[MPI_MAX_PROCESSOR_NAME];
    int namelen;
    MPI::Get_processor_name(name_buffer, namelen);
    assert(namelen >= 0);
    m_name.assign(name_buffer, namelen);
    // Initialize the receiver
    m_receiver.initialize(m_id, &m_sender, &m_handler_thread, tagid);
    m_sender.set_numprocs(m_numprocs);
    m_sender.set_id(m_id);
  }

  size_t mpi_single_tag_handler::id() {
    return m_id;
  }

  std::string mpi_single_tag_handler::
  name() {
    return m_name;
  }

  size_t mpi_single_tag_handler::
  num_processes() {
    return m_numprocs;
  }

  void mpi_single_tag_handler::
  register_handler(int message_type, 
                   mpi_single_tag_handler::po_box_callback* callback) {
    tagid = message_type;
    m_handler_thread.set_callback(callback);
    m_handler_thread.start();
  }

  void mpi_single_tag_handler::
  unregister_handler(int message_type) {
    m_handler_thread.stop();
    m_handler_thread.join();
  }



  void mpi_single_tag_handler::
  start(po_box_callback* callback) {
    register_handler(tagid,callback);
    // Luanch the receiver thread
    m_receiver.start();
    // Launch the sender thread
    m_sender.start();
  }
  
  
  void mpi_single_tag_handler::
  bcast_message(size_t body_size, const char* body, 
                bcast_type bcast) {
      // Create a message object and fill in the necessary fields 
      message msg;
      msg.orig = m_id;
      msg.dest = 0;
      msg.bcast = bcast;
      msg.is_bcast = true;
      msg.type = tagid;
      msg.body_size = body_size;
      msg.body = body;
      // Add the message to the sender queue note that the message
      // memory will be duplicated at this point
      m_sender.add(msg);
  }

  void mpi_single_tag_handler::
  stopAll() {
    std::cout << "Stop Invoked!!!!!!!!!!!!" << std::endl;
   
    message msg;
    msg.orig = m_id;
    msg.dest = 0;
    msg.bcast = ALL;
    msg.is_bcast = true;
    msg.type = tagid;
    msg.body_size = 0;
    msg.body = "";
    msg.quit=true;
    // Add the message to the sender queue note that the message
    // memory will be duplicated at this point
    m_sender.add(msg);
  }


  void mpi_single_tag_handler::
  send_message(size_t dest, 
               size_t body_size, 
               const char* body) {
    // Create a message object and fill in the necessary fields 
    message msg;
    msg.orig = m_id;
    msg.dest = dest;
    msg.is_bcast = false;
    msg.type = tagid;
    msg.body_size = body_size;
    msg.body = body;
    // Ensure that message is valid to send
    assert(msg.dest < m_numprocs);
    // Add the message to the sender queue note that the message
    // memory will be duplicated at this point
    m_sender.add(msg);
  }


  // wait for all threads to terminate before closing
  mpi_single_tag_handler::
  ~mpi_single_tag_handler() {
    // Shut down the mpi connection
    std::cout << "SHUTDWON!" << std::endl;
  }

  void mpi_single_tag_handler::wait() {
    // First wait for all sends to terminate
    m_sender.join();
    // We can't turn off the receiver until everyone has turned off
    // their sender (no active sends waiting to take place)
    try{
      MPI::COMM_WORLD.Barrier();
    } catch( MPI::Exception e) {
      std::cout << "Exception 7: " << e.Get_error_string() << std::endl;
    }
    // Shutdown the receiver
    m_receiver.stop();
    m_receiver.join();
    // Shutdown all the handlers
    m_handler_thread.stop();
    m_handler_thread.join();

    // Finished waiting for the mpi_post_office process
  }


  std::ostream&
  operator<<(std::ostream& out, const mpi_single_tag_handler::message& msg) {
    return out <<"[" 
               << msg.orig << "->" << msg.dest 
               << "(" << msg.type << ") "
               << msg.body_size << " : " << msg.body 
               << "]";
  }

}
#include <sill/macros_undef.hpp>

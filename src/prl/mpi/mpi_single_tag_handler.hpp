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

#ifndef PRL_POST_OFFICE
#define PRL_POST_OFFICE


#include <mpi.h>

#include <list>
#include <string>
#include <cassert>

#include <prl/mpi/mpi_protocols.hpp>
#include <prl/parallel/blocking_queue.hpp>
#include <prl/parallel/pthread_tools.hpp>

namespace prl {


  class mpi_single_tag_handler {

   
    
  public:

    enum bcast_type {ALL, ALL_BUT_SELF};

    //! A very simple message struct
    struct message {
      //! The origin id of the message
      size_t orig;
      //! The destination id of the message
      size_t dest;
      //! The message is being sent for broadcast
      bool is_bcast;
      bcast_type bcast;
      //! The type of a message is a user defined identifier
      int type;
      //! The actual size of the message
      size_t body_size;
      
      //! A pointer to the array of bytes represented by the message
      const  char* body;
      
      bool quit;
      message() : 
        orig(0), dest(0), is_bcast(false), bcast(ALL), 
        type(0), body_size(0), body(NULL),quit(false) { } 
    };

    /**
     * This object is used to manager callback
     */
    class po_box_callback {
    public:
      /**
       * When a message is received by the post office it is first put
       * in a queue for the appropriate box (message type).  Then a
       * second thread (associated with just that box) invokes this
       * method passing the message in as an argument.
       */
      void virtual recv_message(const message& msg) = 0;

      /**
       * This method is invoked once the handler is asked to
       * terminate.
       */
      void virtual terminate() = 0;

      /** User implemented if necessary */ 
      virtual ~po_box_callback() { }

    }; // End of class po_box_callback

  private:
    //! The ID of this mpi_wrapper
    size_t m_id;
    //! The total number of MPI processes
    size_t m_numprocs;
    //! The name of this process
    std::string m_name;

    //! type of the in and out queues
    typedef blocking_queue<message> queue_type;
    
    /**
     * the type of the thread responsible for sending messages.  The
     * sender thread makes a copy of the message when add_to_send is
     * invoked.
     */
    class sender_thread : public thread { 
      bool m_alive;
      queue_type m_queue;
      MPI::Request m_request;
      size_t m_id;
      size_t m_numprocs;
    public:
      sender_thread();
      void set_numprocs(size_t numprocs);
      void set_id(size_t id);
      void add(const message& msg);
      void run();
      void finish();
      ~sender_thread();
    };

    //! thread responsible for sending messages
    sender_thread m_sender;


    //! PO boxes (protocol handlers)
    class handler_thread : public thread {
      po_box_callback* m_callback;
      queue_type m_queue;
      bool m_alive;
    public:
      handler_thread();
      //! Set the callback handler for this thread
      void set_callback(po_box_callback* callback);
      /**
       * The handle method should make a copy of the message locally.
       * This allows the po_box to have a large initial recv buffer
       */
      void add(const message& msg);
      void run();
      void stop();
      ~handler_thread();
    };

    //! a handler function
    handler_thread m_handler_thread;
    
    /**
     * This thread is responsible for receiving messages from mpi and
     * then places the message in the approriate handler.
     */
    class receiver_thread : public thread {
      bool m_alive;
      size_t m_id;
      int m_tagid;
      handler_thread* m_handler;
      sender_thread* m_sender;
      size_t m_body_size;
      char* m_body;
      MPI::Request m_request;
    public:
      receiver_thread();
      void initialize(size_t id,                
                      sender_thread* sender,
                      handler_thread* handler, 
                      int tagid,
                      size_t buffer_size);
      void run(); 
      void stop();
      ~receiver_thread();
    };

    //! thread responsible for receiving messages
    receiver_thread m_receiver;    
    
//     //! Helper routine
//     message duplicate(const message& msg);
//     //! Helper routine
//     void do_send(message& msg, MPI::Request& request);


    /**
     * Register a handler (callback) for a particular message type and
     * then launch that handler thread.
     */
    void register_handler(int type, po_box_callback* callback);
    int tagid;
    /**
     * Register a handler (callback) for a particular message type and
     * then launch that handler thread.
     */
    void unregister_handler(int type);


  public:
    /**
     * Initialize this mpi_wrapper.  MPI requires access to the
     * command line arguments
     */
    mpi_single_tag_handler(int type);
    
  
    //! get the id (rank) of this post master
    size_t id();

    //! get the processor name
    std::string name();

    //! get the total number of mpi processes 
    size_t num_processes();


    /**
     * start all threads.  This should only be invoked once in the beginning after
     * registering all handlers
     */
    void start(po_box_callback* callback);


    /**
     * cause all threads to die.  This should be called when the
     * program is ready to terminate.  Any remaining messages will be
     * flushed to the registerd handlers.
     *
     */
    void stopAll();


     /**
      * This funciton takes the destination post master id along with a
      * pointer to a message and the size of that message.  The message
      * is then copied into a local buffer.  The message will be sent
      * eventually.
      */
    void send_message(size_t dest, 
                      size_t body_size, const  char* body);

    /**
     * Broadcast a message to all nodes
     */
    void bcast_message(
                       size_t body_size, 
                       const char* body,
                       bcast_type bcast = ALL);

      
    /**
     * Causes the calling thread to block until the mpi program
     * finishes and all threads terminate.
     */
    void wait();


    /**
     * waits for all threads to terminate
     */
    ~mpi_single_tag_handler();
    
  }; // end of mpi_single_tag_handler

  std::ostream& 
  operator<<(std::ostream& out, const mpi_single_tag_handler::message& msg);


} // End of namespace prl

#endif





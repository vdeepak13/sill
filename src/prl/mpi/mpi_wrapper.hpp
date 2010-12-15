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
#include <map>
#include <cassert>
#include <stdint.h>
#include <prl/mpi/mpi_protocols.hpp>
#include <prl/parallel/blocking_queue.hpp>
#include <prl/parallel/pthread_tools.hpp>

namespace prl {
  /**
    This tries to be an wrapper around MPI to provide around the same
    level of functionality as TCP. The MPI post office creates 2 background
    threads, one to perform sends, and the other to perform receives.

    Each message sent and received are tagged with a "protocol ID". The
    protocol ID is used to identify the way the message is handled at the
    destination. The receiver must "register" a message handler with a
    particular protocol ID before messages tagged to that protocol ID can be
    received. Protocol ID's <8 are reserved.

    For instance, if the receiver registers protocol 10 with function A;
    function A will be called with all received messages tagged as protocol 10.

    There are 2 handler types: callback handlers, and queued handlers.
    The function register_handler(PROTOCOL, callbackfunction) is used to
    register a protocol handler.

    A callback handler is created by calling register_handler with a functor
    inherited from mpi_post_office::po_box_callback. The recv_message function
    in the functor will be called with the message upon receipt of a message
    matching the protocol ID, Do note that the recv_message is called within
    the main loop of the receiver thread, so it is not a good idea to perform
    extensive processing within this function. If extensive processing is
    needed, the function should queue it up for processing by another thread.
    Upon termination of the mpi_post_office, or deregistration of the handler,
    the terminate() function will be called.
    
    A queued handler is created by calling register_handler with NULL in the
    second argument. When a message is received matching the protocol ID, the
    message will be queued within the mpi_post_office. Calling receive() or
    async_receive() will read the message from the queue.

    You can only register 1 handler per message type.

    There is a special class of handlers called "condition variable handlers"
    which can be created and deleted by register_condvar() and
    unregister_condvar() respectively. These simply call signal()
    on the registered condition variable once a message with a matching
    protocol ID is received. You can register a condition variable with a
    protocol ID of "-1" which will trigger on all non-control messages
    (IDs >= 8)

    You can have multiple condition variable handlers per protocol ID, and
    condition variable handlers can coexist with callback handlers and queued
    handlers. We guarantee that the condition variable will only be signaled
    after the regular handlers are executed. (i.e., you can use the condition
    variable handler to perform an asynchronous wait on the queued handler)
    
    Race Conditions:
    There is a race condition which may be met on stopping the post office.
    This is due to an interaction between MPI_Cancel and MPI_Wait.
    If MPI_Cancel is called, but MPI_Wait completes successfully, the 
    cancellation will attempt to cancel a NULL request, resulting in a failure.
    This is a known issue in MPI 2.0. I have tried to reduce the race window
    as much as possible, but there is a still a chance that a crash may occur on
    termination. It is therefore recommended to stop() the post office as late
    in the execution as possible (after all output is complete) so that there is
    no data loss on termination.
  */
  class mpi_post_office {
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
        /** If this flag is set(non zero), the transmission of this message will not
        increment num_msg_sent or num_msg_recv (on the receiver's side)
        This is useful for transmitting true "background" messages that
        do not contribute towards the consensus protocol*/
        bool uncounted;
        //! The actual size of the message
        size_t body_size;
        //! A pointer to the array of bytes represented by the message
        const  char* body;
        message() :
          orig(0), dest(0), is_bcast(false), bcast(ALL),
          type(0), uncounted(false), body_size(0), body(NULL) { }

        message duplicate() const {
          // Make a copy of this original message
          message new_msg = *this;
          // Make a copy of the buffer
          char* newbody = new  char[this->body_size];
          assert(newbody != NULL);
          new_msg.body_size = this->body_size;
          // do a memory copy
          memcpy(newbody, this->body, this->body_size);
          // Make the new message body the new body;
          new_msg.body = newbody;
          // Return a new message which is a duplicate the original
          return new_msg;
        }
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
          void virtual terminate() {};
          /** User implemented if necessary */
          virtual ~po_box_callback() { }

      }; // End of class po_box_callback

    private:

      /**
      * the type of the thread responsible for sending messages.  The
      * sender thread makes a copy of the message when add_to_send is
      * invoked.
      */
      class sender_thread : public thread {
        private:
          bool m_alive;
          blocking_queue<message> m_queue;
          MPI_Request m_request;
          size_t m_id;
          size_t m_numprocs;
        public:
          sender_thread();
          void set_numprocs(size_t numprocs);
          void set_id(size_t id);
          void add(const message& msg);
          void run();
          void finish();
          void flush();
          ~sender_thread();
      };

    /**
      * This thread is responsible for receiving messages from mpi and
      * then places the message in the approriate handler.
      */
      class receiver_thread : public thread {
        private:
          bool m_alive;
          bool m_receiving;
          
          size_t m_id;
          std::map<int, po_box_callback*>* m_handlers;
          sender_thread* m_sender;
          size_t m_body_size;
          char* m_body;
          MPI_Request m_request;
          mpi_post_office *m_po;
          spinlock m_statuslock;
        public:
          receiver_thread();
          void initialize(size_t id,
                          sender_thread* sender,
                          std::map<int, po_box_callback*>* handlers,
                          mpi_post_office *po,
                          size_t buffer_size);
          void run();
          void stop();
          ~receiver_thread();
      };

      //! The ID of this MPI node
      size_t m_id;

      //! The total number of MPI processes
      size_t m_numprocs;

      //! number of packets sent and received
      size_t numsent, numrecv;

      //! number of control packets sent and received
      size_t ctrlsent, ctrlrecv;


      //! thread responsible for sending messages
      sender_thread m_sender;

      spinlock handler_lock;
      std::map<int, po_box_callback*> m_handlers;
      std::multimap<int, conditional*> m_condvar_handlers;
      std::map<int, blocking_queue<message>* > m_syncreceives;


      //! thread responsible for receiving messages
      receiver_thread m_receiver;

  //     //! Helper routine
  //     message duplicate(const message& msg);
  //     //! Helper routine
  //     void do_send(message& msg, MPI::Request& request);


    public:
      /**
      * Initialize this mpi_wrapper.  MPI requires access to the
      * command line arguments
      */
      mpi_post_office();


      //! get the id (rank) of this post master
      size_t id();

      //! get the total number of mpi processes
      size_t num_processes();

      //! get the total number of messages sent
      size_t num_msg_sent();
      //! get the total number of messages received
      size_t num_msg_recv();

      //! get the total number of control messages sent
      size_t num_ctrl_sent();
      //! get the total number of control messages received
      size_t num_ctrl_recv();

      /**
      * Register a handler (callback) for a particular message type and
      * then launch that handler thread. If callback is NULL, the message will
      * be buffered in a queue instead of calling a handler. The message can then
      * be read by calling read() or async_read()
      */
      void register_handler(int type, po_box_callback* callback);

      /**
      * Register a handler (callback) for a particular message type and
      * then launch that handler thread.
      */
      void unregister_handler(int type);

      /**
      * Registers a condition variable that is signalled if a message type is
      * matched. If type is == -1, all non-control messages will signal the condvar
      */
      void register_condvar(int type, conditional* cond);

      /**
      * Deregisters a condition variable which was registered with register_condvar
      */
      void unregister_condvar(conditional* cond);

      /// Reads a message from a message queue. Blocking. Return true on success.
      bool receive(size_t type, message &msg);

      /**
      *  Reads a message from a message queue. Non-blocking.
      *  Return true on success.
      */
      bool async_receive(size_t type, message &msg);

      /**
      * start all threads.  This should only be invoked once in the beginning after
      * registering all handlers
      */
      void start();


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
      void send_message(size_t dest, int type,
                        size_t body_size, const  char* body,
                        bool uncounted = false);

      /**
        Flushes the send queue
      */
      void flush();
      /**
      * Broadcast a message to all nodes
      */
      void bcast_message(int type,
                        size_t body_size,
                        const char* body,
                        bcast_type bcast = ALL);


      /**
      * simple barrier
      */
      void barrier();



      /**
      * Causes the calling thread to block until the mpi program
      * finishes and all threads terminate.
      */
      void wait();


      /**
      * waits for all threads to terminate
      */
      ~mpi_post_office();
    
  }; // end of mpi_post_office

  std::ostream&
  operator<<(std::ostream& out, const mpi_post_office::message& msg);


  void mpi_inplace_reduce_uint64(uint64_t &x);
  void mpi_inplace_reduce_uint32(uint32_t &x);

} // End of namespace prl

#endif





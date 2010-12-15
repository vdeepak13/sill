#ifndef MESSAGE_DATA_HPP
#define MESSAGE_DATA_HPP


#include <prl/parallel/pthread_tools.hpp>
#include <prl/factor/norms.hpp>
#include <prl/parallel/object_allocator.hpp>

// This include should always be last
#include <prl/macros_def.hpp>

namespace prl {
  /**
   * \todo move this into the basic_state_manager.
   */
  enum ReadWrite {Reading, Writing};

  /**
   * \class MessageData
   *
   * Handles the synchronizing (checkin and checkout) of a message
   * 
   * This class implements a kind of reader/writer semantics, but take
   * the assumption that there can only be single writer at any one
   * time.  (assertion failure if two writers try to grab it)
   *  
   * checkouts for reads will give it the original message, but
   * checkouts for writes will provide a writebuffer (which will have
   * the same contents).  When the writebuffer is checked in, it will
   * be "committed" by updating the message with the writebuffer.
   *
   * A "pool" object has to be provided to allocate the write buffers
   *
   * \todo move this into the basic state manager.
   */
  template<typename F>
  class MessageData {
  public:
    F message;
    spinlock lock;  // Locks the write buffer and the readercount
    F *writebuffer;
    char readercount;
    double lastresidual;
    MessageData() : message(1.0) {
      writebuffer = NULL;
      readercount = 0;
      lastresidual = 100;
    }

    /**
     * Checks the message.  If its a write request, a buffer will be
     * taken from the object_allocator
     */
    F *checkout(object_allocator_tls<F> &pool, 
                const ReadWrite rw) {
      if (rw == Reading) {
        lock.lock();
        readercount++;
        lock.unlock();
        return &message;
      }
      else {
        lock.lock();
        // we only support 1 writer. 
        assert(writebuffer == NULL);
        writebuffer = pool.checkout();
        lock.unlock();
        (*writebuffer) = message;

        return writebuffer;
      }
    }

    /**
     * Checks the message.  If its a write request, a buffer will be
     * taken from the object_allocator.  If the object is currently
     * allocated for writing then this function will return NULL
     */
    F *trycheckout(object_allocator_tls<F> &pool, 
                const ReadWrite rw) {
      if (rw == Reading) {
        return checkout(pool,rw);
      }
      else {
        lock.lock();
        if (writebuffer != NULL) {
          lock.unlock();
          return NULL;
        }
        writebuffer = pool.checkout();
        lock.unlock();
        (*writebuffer) = message;
        return writebuffer;
      }
    }

    /**
     * Checks in the message.  If the message was checked out for
     * writing, the buffer will be committed and returned to the
     * object_allocator if allow_simultaneous_rw is not set, the
     * buffer will be committing regardless of whether there is an
     * existing reader.  If it is set, the thread will wait until
     * there are no readers before committing the buffer.  Returns the
     * norm of the change of value. (using the norm() function) if its
     * a write checkin. Returns 0 otherwise.
     *
     * \warning writes may block for extended periods if there are
     * readers
     */
    double checkin(object_allocator_tls<F> &pool, 
                   const F *msg, const factor_norm<F>& norm, 
                   const bool allow_simultaneous_rw) {
      double residual = -1;
      lock.lock();
      // if it matches message address, this was a read request
      if (msg == &message) {
        assert(readercount > 0);
        readercount--;
      }
      //this was a write request
      else if (msg == writebuffer) {
        //Optional: Wait for readers to complete
        if (!allow_simultaneous_rw) {
          while(readercount > 0) {
            lock.unlock();
            sched_yield();
            lock.lock();
          }
        }
        // compute the delta
        residual = norm(message , *(writebuffer));
        message = *(writebuffer);
        pool.checkin(writebuffer);
        writebuffer = NULL;
        lastresidual = residual;
      }
      else {
        //Erroneous message!
        assert(0);
      }
      lock.unlock();
      return residual;
    }
  }; // End of message_data



  /**
   * \class MessageDataUnsynchronized
   *
   * Handles the synchronizing (checkin and checkout) of a message
   * 
   * This class implements a kind of reader/writer semantics
   * This class is only partially synchronized. and provides semi protection 
   * from simultaneous reading and writing. This allows for multiple 
   * simultaneous checkouts for writing however.
   *
   * This class should be used only if no simultaneous writes are
   * guaranteed (or that it is known to be safe)
   * \todo move this into the basic state manager.
   */
  template<typename F>
  class MessageDataUnsynchronized {
  public:
    F message;
    double lastresidual;
//    mutex t;
    MessageDataUnsynchronized() : message(1.0) {
    
    }


    F *checkout(multi_object_allocator_tls<F> &pool, const ReadWrite rw) {
      if (rw == Writing) {
        F* writebuffer = pool.checkout();
        (*writebuffer) = message;
        return writebuffer;
      }
      else {
//        t.lock();
        return &message;
      }
    }


    F *trycheckout(multi_object_allocator_tls<F> &pool, const ReadWrite rw) {
      return checkout(rw);
    }

    double checkin(multi_object_allocator_tls<F> &pool, const F *msg, 
                   const factor_norm<F>& norm,
                   const bool allow_simultaneous_rw /*unused*/) {
      double residual = -1;
      //this was a write request
      if (msg != &message){
        residual = norm(message , *(msg));
//        if (allow_simultaneous_rw) t.lock();
        message = *(msg);
        pool.checkin(msg);
        lastresidual = residual;
//        if (allow_simultaneous_rw) t.unlock();
      }
      else {
//        t.unlock();
      }
      return residual;
    }
  }; // End of message_data

} //end of prl namespace
#include <prl/macros_undef.hpp>

#endif

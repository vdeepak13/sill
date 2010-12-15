#ifndef PRL_DISTRIBUTED_AVERAGING_HPP
#define PRL_DISTRIBUTED_AVERAGING_HPP

#include <boost/thread/mutex.hpp>

#include <prl/global.hpp>
#include <prl/serialization/serialize.hpp>

namespace prl {

  template <typename T>
  class distributed_averaging {

    //! The number of transmitted bytes
    uint64_t bytes_in_;

    //! The number of received bytes
    uint64_t bytes_out_;

    //! The corresponding mutex
    mutable boost::mutex bytes_mutex;

  public:
    distributed_averaging() 
      : bytes_in_(), bytes_out_() {}

    virtual ~distributed_averaging() { }

    //! Returns the bytes sent
    uint64_t bytes_out() const {
      boost::mutex::scoped_lock lock(bytes_mutex);
      return bytes_out_;
    }

    //! Returns the bytes received
    uint64_t bytes_in() const {
      boost::mutex::scoped_lock lock(bytes_mutex);
      return bytes_in_;
    }
    
    //! Locks the mutex on the value
    virtual void lock() const = 0;

    //! Unlocks the mutex on the value
    virtual void unlock() const = 0;

    //! Returns the current estimate of the average value. Does not lock.
    virtual const T& value() const = 0;

    //! Sets the current estimate of the average value. Does not lock.
    virtual void set_value(const T& value) = 0;

    //! Perform one iteration 
    virtual void iterate() = 0;

    //! Wait for the specified amount of time in seconds (in expectation)
    virtual void sleep(size_t delay) = 0;
    
  protected:
    //! Adds the received bytes
    void add_bytes(const iarchive& ar) {
      boost::mutex::scoped_lock lock(bytes_mutex);
      bytes_in_ += ar.bytes();
    }

    //! Adds the sent bytes
    void add_bytes(const oarchive& ar) {
      boost::mutex::scoped_lock lock(bytes_mutex);
      bytes_out_ += ar.bytes();
    }

  }; // class distributed_averaging

}

#endif

#ifndef PRL_SYNCHRONOUS_GOSSIP_TIMED_HPP
#define PRL_SYNCHRONOUS_GOSSIP_TIMED_HPP

#include <iostream>

#include <boost/asio.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/shared_ptr.hpp>

#include <prl/global.hpp>
#include <prl/serialization/serialize.hpp>
#include <prl/distributed/random_host.hpp>
#include <prl/distributed/distributed_averaging.hpp>

namespace prl {

  /**
   * A class that implements a synchronous gossip algorithm based on
   * Boyd, Ghosh, Prabhakar, Shah: Randomized Gossip Algorithms
   * IEEE Transactions on Information Theory, Vol 52, No 6, June 2006
   * with time-outs for the operations.
   */
  template <typename T>
  class synchronous_gossip_timed : public distributed_averaging<T> {

  private:
    //! Mutex to access the value
    mutable boost::mutex mutex;

    //! The IO
    boost::asio::io_service io_service;

    //! The current estimate 
    T value_;

    //! The server's hostname
    std::string hostname;

    //! The server's port
    unsigned short port;

    //! A pointer to function object that returns random hosts
    random_host* hostfn;

    //! The timeout time for server/client operations (0 means no limit)
    long timeout;

    //! Is the node currently active?
    bool active;

    //! The corresponding mutex
    boost::mutex active_mutex;

    //! And the condition variable
    boost::condition_variable active_condition;

    struct interrupted_flag {
      interrupted_flag() : flag(false), running(true) { }
      bool flag;
      bool running;
      boost::mutex mutex;
    };

    typedef boost::shared_ptr<interrupted_flag> interrupted_flag_ptr;

    typedef boost::asio::ip::tcp tcp;
    
  public:
    //! Constructs the protocl, starting with the specified value
    synchronous_gossip_timed(const T& value,
			     const std::string& hostname,
			     unsigned short port,
			     random_host* hostfn,
			     long timeout) 
      : value_(value),
        hostname(hostname),
        port(port),
        hostfn(hostfn),
	timeout(timeout),
	active(false) {
      boost::thread thread(boost::bind(&synchronous_gossip_timed::server,this));
    }

    // distributed_averaging interface
    void lock() const {
      mutex.lock();
    }

    void unlock() const {
      mutex.unlock();
    }
    
    const T& value() const {
      return value_;
    }

    void set_value(const T& value) {
      value_ = value;
    }

    void iterate() {
      client();
    }

    void sleep(size_t delay) {
      boost::this_thread::sleep(boost::posix_time::seconds(delay));
    }

    //! A single session of the server
    void server_session(tcp::iostream* stream, 
			interrupted_flag_ptr interrupted) {
      // debugging info
      using namespace std;
      std::string clientname;
      getline(*stream, clientname);
      cout << "server: contacted by " << clientname << endl;

      try {
	// receive the half-value from the client
	T recv_half;
	iarchive iar(*stream);
	iar >> recv_half;
	this->add_bytes(iar);

	// compute the local average
	T sent_half;
	interrupted->mutex.lock();
	if (interrupted->flag) {
	  interrupted->mutex.unlock();
	  delete stream;
	  return;
	} else {
	  lock();
	  value_ /= 2;
	  sent_half = value_;
	  value_ += recv_half;
	  unlock();
	  interrupted->mutex.unlock();
	}

	// send the half-value to the client
	// note: if the second phase aborts, we will no longer have
	// a consistent sum across nodes; ignore this for now
	oarchive oar(*stream);
	oar << sent_half;
	this->add_bytes(oar);

	cout << "server: completed " << clientname << endl;
      } catch(std::exception& e) {
	cout << "server: aborted " << clientname << endl;
	cout << "        " << e.what() << endl;
      }
      delete stream;
      interrupted->mutex.lock();
      interrupted->running = false;
      interrupted->mutex.unlock();
    }

    void server_timer(tcp::iostream* stream) {
      interrupted_flag_ptr interrupted(new interrupted_flag());
      boost::thread thread(boost::bind(&synchronous_gossip_timed::server_session,
				       this, stream, interrupted));
      if (timeout > 0)
	thread.timed_join(boost::posix_time::seconds(timeout));
      else
	thread.join();
      boost::mutex::scoped_lock lock(interrupted->mutex);
      interrupted->flag = true;
      if (interrupted->running) {
 	using namespace std;
	cout << "server: interrupted" << endl;
      }
      active_mutex.lock();
      active = false;
      active_mutex.unlock();
      active_condition.notify_one();
    }

    //! The server side of the implementation
    void server() {
      tcp::endpoint endpoint(tcp::v4(), port);
      tcp::acceptor acceptor(io_service, endpoint);
      while(1) {
        tcp::iostream* stream = new tcp::iostream;
        acceptor.accept(*stream->rdbuf());
	boost::mutex::scoped_lock lock(active_mutex);
	if (!active) {
	  active = true;
	  boost::thread thread(boost::bind(&synchronous_gossip_timed::server_timer,
					   this, stream));
	} else {
	  // drop the connection
	  delete stream;
	}
      }
    }

    bool client_session(const std::string& hostname2,
			unsigned short port2, 
			interrupted_flag_ptr interrupted) {
      // open the connection
      tcp::iostream stream(hostname2, boost::lexical_cast<std::string>(port2));
      if (!stream) {
// 	boost::mutex::scoped_lock lock(active_mutex);
// 	active = false;
	interrupted->mutex.lock();
	interrupted->running = false;
	interrupted->mutex.unlock();
	return false;
      }

      // debug info
      using namespace std;
      stream << hostname << std::endl;
      cout << "client: contacted " << hostname2 << endl;

      // locking is not necessary since no new server threads are activated 
      bool success;
      T half;
      interrupted->mutex.lock();
      if (interrupted->flag) {
	interrupted->mutex.unlock();
	return false;
      } else {
	lock();
	half = value_;
	unlock();
	half /= 2;
	interrupted->mutex.unlock();
      }

      try {
	// send a half to the server
	oarchive oar(stream);
	oar << half;
	this->add_bytes(oar);

	interrupted->mutex.lock();
	if (interrupted->flag) {
	  interrupted->mutex.unlock();
	  return false;
	} else {
	  interrupted->mutex.unlock();
	}

	// receive a half from the server
	iarchive iar(stream);
	iar >> half;
	this->add_bytes(iar);

	interrupted->mutex.lock();
	if (interrupted->flag) {
	  interrupted->mutex.unlock();
	} else {
	  // update the local model
	  // lock just in case something bad happens
	  lock();
	  value_ /= 2;
	  value_ += half;
	  unlock();
	  interrupted->mutex.unlock();
	}

	cout << "client: completed " << hostname2 << endl;
	success = true;
      } catch (std::exception& e) { 
	cout << "client: aborted " << hostname2 << endl;
	cout << "        " << e.what() << endl;
        success = false;
      }

      interrupted->mutex.lock();
      interrupted->running = false;
      interrupted->mutex.unlock();
      return success;
    }

    //! The client side of the implementation
    void client() {
      // With probability 1/2, do nothing
      if (rand() % 2)
	return;
      
      // generate a random connection
      std::string hostname2;
      unsigned short port2;
      try {
	boost::tie(hostname2, port2) = (*hostfn)();
      } catch(std::exception& e) {
	return;
      }
      if (hostname == hostname2 && port == port2)
        return; // noop

      // wait for any server sessions to finish and then activate
      {
	boost::mutex::scoped_lock lock(active_mutex);
	while (active) 
	  active_condition.wait(lock);
	active = true;
      }

      // implement timeout
      interrupted_flag_ptr interrupted(new interrupted_flag());
      boost::thread thread(boost::bind(&synchronous_gossip_timed::client_session,
				       this, hostname2, port2, interrupted));
      bool completed = true;
      if (timeout > 0)
	completed = thread.timed_join(boost::posix_time::seconds(timeout));
      else
	thread.join();

      {
	boost::mutex::scoped_lock lock(interrupted->mutex);
	interrupted->flag = true;
	if (interrupted->running) {
	  using namespace std;
	  cout << "client: interrupted" << endl;
	}
      }
      
      // deactivate
      boost::mutex::scoped_lock lock(active_mutex);
      active = false;
    }
    
  }; // class synchronous_gossip_timed

}

#endif

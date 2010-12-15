#ifndef PRL_SYNCHRONOUS_GOSSIP_HPP
#define PRL_SYNCHRONOUS_GOSSIP_HPP

#include <iostream>

#include <boost/asio.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>

#include <prl/global.hpp>
#include <prl/serialization/serialize.hpp>
#include <prl/distributed/random_host.hpp>
#include <prl/distributed/distributed_averaging.hpp>

namespace prl {

  /**
   * A class that implements a synchronous gossip algorithm based on
   * Boyd, Ghosh, Prabhakar, Shah: Randomized Gossip Algorithms
   * IEEE Transactions on Information Theory, Vol 52, No 6, June 2006
   */
  template <typename T>
  class synchronous_gossip : public distributed_averaging<T> {

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

    //! Is the node currently active?
    bool active;

    //! The corresponding mutex
    boost::mutex active_mutex;

    //! And the condition variable
    boost::condition_variable active_condition;

    typedef boost::asio::ip::tcp tcp;
    
  public:
    //! Constructs the protocl, starting with the specified value
    synchronous_gossip(const T& value,
		       const std::string& hostname,
		       unsigned short port,
		       random_host* hostfn) 
      : value_(value),
        hostname(hostname),
        port(port),
        hostfn(hostfn),
	active() {
      boost::thread thread(boost::bind(&synchronous_gossip::server, this));
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
    void server_session(tcp::iostream* stream) {
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
	lock();
	value_ /= 2;
	T sent_half = value_;
	value_ += recv_half;
	unlock();

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
	  boost::thread thread(boost::bind(&synchronous_gossip::server_session,
					   this, stream));
	} else {
	  // drop the connection
	  delete stream;
	}
      }
    }

    //! The client side of the implementation
    bool client() {
      // With probability 1/2, do nothing
      if (rand() % 2)
	return false;
      
      // generate a random connection
      std::string hostname2;
      unsigned short port2;
      try {
	boost::tie(hostname2, port2) = (*hostfn)();
      } catch(std::exception& e) {
	return false;
      }
      if (hostname == hostname2 && port == port2)
        return true; // noop

      // wait for any server sessions to finish and then activate
      {
	boost::mutex::scoped_lock lock(active_mutex);
	while (active) 
	  active_condition.wait(lock);
	active = true;
      }

      // open the connection
      tcp::iostream stream(hostname2, boost::lexical_cast<std::string>(port2));
      if (!stream) {
	boost::mutex::scoped_lock lock(active_mutex);
	active = false;
	return false;
      }

      // debug info
      using namespace std;
      stream << hostname << std::endl;
      cout << "client: contacted " << hostname2 << endl;

      // perform strict locking
      bool success;
      lock();
      T half = value_;
      half /= 2;

      try {
	// send a half to the server
	oarchive oar(stream);
	oar << half;
	this->add_bytes(oar);

	// receive a half from the server
	iarchive iar(stream);
	iar >> half;
	this->add_bytes(iar);

	// update the local model
	value_ /= 2;
	value_ += half;

	cout << "client: completed " << hostname2 << endl;
	success = true;
      } catch (std::exception& e) { 
	cout << "client: aborted " << hostname2 << endl;
	cout << "        " << e.what() << endl;
        success = false;
      }
      
      unlock();
      
      // deactivate
      boost::mutex::scoped_lock lock(active_mutex);
      active = false;

      return success;
    }
    
  }; // class synchronous_gossip

}

#endif

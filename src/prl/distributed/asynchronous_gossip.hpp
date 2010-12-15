#ifndef PRL_ASYNCHRONOUS_GOSSIP_HPP
#define PRL_ASYNCHRONOUS_GOSSIP_HPP

#include <iostream>

#include <boost/asio.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/random/lagged_fibonacci.hpp>
#include <boost/random/exponential_distribution.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

#include <prl/global.hpp>
#include <prl/serialization/serialize.hpp>
#include <prl/distributed/random_host.hpp>
#include <prl/distributed/distributed_averaging.hpp>

namespace prl {

  /**
   * A class that implements the asynchronous gossip algorithm from
   * Boyd, Ghosh, Prabhakar, Shah: Randomized Gossip Algorithms
   * IEEE Transactions on Information Theory, Vol 52, No 6, June 2006
   */
  template <typename T>
  class asynchronous_gossip : public distributed_averaging<T> {

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

    //! Implement strict locking?
    bool strict;

    //! Maximum number of active server threads
    size_t maxload;

    //! Number of active server threads
    size_t nthreads;

    //! The corresponding mutex
    boost::mutex nthreads_mutex;

    //! The random number generator; only used in iterate()
    boost::lagged_fibonacci607 rng;

    typedef boost::asio::ip::tcp tcp;
    
  public:
    //! Constructs the protocl, starting with the specified value
    asynchronous_gossip(const T& value,
			const std::string& hostname,
			unsigned short port,
			random_host* hostfn,
			bool strict,
			size_t maxload) 
      : value_(value),
        hostname(hostname),
        port(port),
        hostfn(hostfn),
	strict(strict),
	maxload(maxload),
	nthreads() {
      boost::thread thread(boost::bind(&asynchronous_gossip::server, this));
      using namespace boost::posix_time;
      time_duration time = microsec_clock::local_time().time_of_day();
      rng.seed(uint32_t(time.total_microseconds()));
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
      boost::exponential_distribution<> p(1.0/delay);
      long ms = (long)(p(rng) * 1000);
      boost::this_thread::sleep(boost::posix_time::milliseconds(ms));
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
      boost::mutex::scoped_lock lock(nthreads_mutex);
      nthreads--;
    }

    //! The server side of the implementation
    void server() {
      tcp::endpoint endpoint(tcp::v4(), port);
      tcp::acceptor acceptor(io_service, endpoint);
      while(1) {
        tcp::iostream* stream = new tcp::iostream;
        acceptor.accept(*stream->rdbuf());
	boost::mutex::scoped_lock lock(nthreads_mutex);
	if (nthreads < maxload) {
	  nthreads++;
	  boost::thread thread(boost::bind(&asynchronous_gossip::server_session,
					   this, stream));
	} else {
	  // drop the connection
	  delete stream;
	}
      }
    }

    //! The client side of the implementation
    bool client() {
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

      // open the connection
      tcp::iostream stream(hostname2, boost::lexical_cast<std::string>(port2));
      if (!stream) return false;

      // debug info
      using namespace std;
      stream << hostname << std::endl;
      cout << "client: contacted " << hostname2 << endl;

      // this part implements both strict and loose locking
      T half;
      lock();
      half = value_;
      half /= 2;
      if (!strict) unlock();

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
	if (!strict) lock();
	value_ /= 2;
	value_ += half;
	unlock();

	cout << "client: completed " << hostname2 << endl;
        return true;
      } catch (std::exception& e) { 
	if (strict) unlock();
	cout << "client: aborted " << hostname2 << endl;
	cout << "        " << e.what() << endl;
        return false; // ignore an unsuccessful attempt
      }
    }
    
  }; // class asynchronous_gossip

}

#endif

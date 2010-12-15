#ifndef PRL_PUSHSUM_AVERAGING_HPP
#define PRL_PUSHSUM_AVERAGING_HPP

#include <iostream>

#include <boost/asio.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>

#include <prl/global.hpp>
#include <prl/serialization/serialize.hpp>
#include <prl/distributed/random_host.hpp>
#include <prl/distributed/distributed_averaging.hpp>

namespace prl {


  /**
   * A class that implements the push-sum protocol for averaging with random 
   * nodes in the network. 
   */
  template <typename T>
  class pushsum_averaging : public distributed_averaging<T> {

  private:
    //! Mutex to access total
    mutable boost::mutex mutex;

    //! The IO
    boost::asio::io_service io_service;

    //! The current estimate 
    T total;

    //! The current weight
    double weight;

    //! Cached value (so that value() can return a reference)
    mutable T value_;

    //! The server's hostname
    std::string hostname;

    //! The server's port
    unsigned short port;

    //! A pointer to function object that returns random hosts
    random_host* hostfn;

    //! If true, the server sends half of its value/weight to the client
    bool symmetric;

    //! Maximum number of active server threads
    size_t maxload;

    //! Number of active server threads
    size_t nthreads;

    //! The corresponding mutex
    boost::mutex nthreads_mutex;

    //! The number of transmitted bytes
    uint64_t bytes_sent_;

    //! The number of received bytes
    uint64_t bytes_rcvd_;

    //! The corresponding mutex
    mutable boost::mutex bytes_mutex;

    typedef boost::asio::ip::tcp tcp;
    
  public:
    //! Constructs the protocl, starting with the specified value
    pushsum_averaging(const T& value,
		      const std::string& hostname,
		      unsigned short port,
		      random_host* hostfn,
		      bool symmetric,
		      size_t maxload) 
      : total(value),
	weight(1),
        hostname(hostname),
        port(port),
        hostfn(hostfn),
	symmetric(symmetric),
	maxload(maxload),
	nthreads(),
	bytes_sent_(),
	bytes_rcvd_() {
      boost::thread thread(boost::bind(&pushsum_averaging::server, this));
    }

    // distributed_averaging interface
    void lock() const {
      mutex.lock();
    }

    void unlock() const {
      mutex.unlock();
    }
    
    const T& value() const {
      value_ = total;
      value_ /= weight;
      return value_;
    }

    void set_value(const T& value) {
      total = value;
      total *= weight;
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

      // receive the value and the associated weight
      T total2;
      double weight2;
      try {
	iarchive iar(*stream);
	iar >> total2;
	iar >> weight2;
	this->add_bytes(iar);
	if (symmetric) {
	  // send half of our weight to the client
	  lock();
	  total /= 2;
	  weight /= 2;
	  T total3 = total;
	  double weight3 = weight;
	  unlock();
	  oarchive oar(*stream);
	  oar << total3;
	  oar << weight3;
	  this->add_bytes(oar);
	}
	// if successful, add them to the local storage
	boost::mutex::scoped_lock lock(mutex);
	total += total2;
	weight += weight2;
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
	  boost::thread thread(boost::bind(&pushsum_averaging::server_session,
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

      // keep a half of the weighted value
      lock();
      total /= 2;
      weight /= 2;
      T total2 = total;
      double weight2 = weight;
      unlock();

      // send the other half to the random node
      try {
	oarchive oar(stream);
	oar << total2 << weight2;
	this->add_bytes(oar);
	if (symmetric) {
	  // receive a total/weight from the server
	  iarchive iar(stream);
	  iar >> total2 >> weight2;
	  lock();
	  total += total2;
	  weight += weight2;
	  unlock();
	  this->add_bytes(iar);
	}
	cout << "client: completed " << hostname2 << endl;
        return true;
      } catch (std::exception& e) { 
	cout << "client: aborted " << hostname2 << endl;
	cout << "        " << e.what() << endl;
        return false; // ignore an unsuccessful attempt
      } 
    }
    
  }; // class pushsum_averaging

}

#endif

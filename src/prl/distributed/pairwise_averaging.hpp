#ifndef PRL_PAIRWISE_AVERAGING_HPP
#define PRL_PAIRWISE_AVERAGING_HPP

#include <iostream>

#include <boost/asio.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>

#include <prl/global.hpp>
#include <prl/serialization/serialize.hpp>
#include <prl/distributed/random_host.hpp>

namespace prl {

  template <typename InternetProtocol>
  std::ostream& operator<<(std::ostream& out, 
                           boost::asio::ip::basic_endpoint<InternetProtocol>& e)
  {
    out << e.address().to_string() << ":" << e.port();
    return out;
  }

  /**
   * A class that implements a pairwise averaging protocol with random 
   * nodes in the network. 
   *
   * TODO: document ordering, add robustness
   *
   * The user needs to overload random_neighbor function.
   * @see list_pairwise_averaging, chord_pairwise_averaging
   */
  template <typename T>
  class pairwise_averaging {

  private:
    //! Mutex to access value_
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

    //! Server's thread
    boost::thread thread;

    //! The number of transmitted bytes
    uint64_t bytes_sent_;

    //! The numberof received bytes
    uint64_t bytes_rcvd_;

    //! The mutex corresponding mutex
    mutable boost::mutex bytes_mutex;

    typedef boost::asio::ip::tcp tcp;
    
  public:
    //! Constructs the protocl, starting with the specified value
    pairwise_averaging(const T& value,
                       const std::string& hostname,
                       unsigned short port,
                       random_host* hostfn) 
      : value_(value),
        hostname(hostname),
        port(port),
        hostfn(hostfn),
        // mutex and io_service are initialized by now and can be used by thread
        thread(boost::bind(&pairwise_averaging::server, this)),
	bytes_sent_(),
	bytes_rcvd_() { }

    //! Locks the mutex on the value
    void lock() const {
      mutex.lock();
    }

    //! Unlocks the mutex on the value
    void unlock() const {
      mutex.unlock();
    }
    
    //! Returns the current estimate of the average value. Does not lock.
    const T& value() const {
      return value_;
    }

    //! Sets the current estimate of the average value. Does not lock.
    void set_value(const T& value) {
      value_ = value;
    }

    //! Adds the received bytes
    void add_bytes(const iarchive& ar) {
      boost::mutex::scoped_lock lock(bytes_mutex);
      bytes_rcvd_ += ar.bytes();
    }

    //! Adds the sent bytes
    void add_bytes(const oarchive& ar) {
      boost::mutex::scoped_lock lock(bytes_mutex);
      bytes_sent_ += ar.bytes();
    }

    //! Returns the bytes sent
    uint64_t bytes_sent() const {
      boost::mutex::scoped_lock lock(bytes_mutex);
      return bytes_sent_;
    }

    //! Returns the bytes received
    uint64_t bytes_received() const {
      boost::mutex::scoped_lock lock(bytes_mutex);
      return bytes_rcvd_;
    }

    //! Receives the value, performs the averaging, and sends the value back
    void perform_averaging(std::iostream& stream) {
      oarchive oar(stream);
      iarchive iar(stream);
      T other;
      iar >> other;
      add_bytes(iar);
      // it is vital that we do not lock until the read succeeds
      boost::mutex::scoped_lock lock(mutex);
      value_ += other;
      value_ /= 2;
      oar << value_;
      add_bytes(oar);
      
    }

    //! Sends the value and waits to receive the average
    void assist_averaging(std::iostream& stream) {
      oarchive oar(stream);
      iarchive iar(stream);
      boost::mutex::scoped_lock lock(mutex);
      oar << value_;
      add_bytes(oar);
      T other;
      iar >> other;
      add_bytes(iar);
      value_ = other;
      // if the >> operation interrupts, we will retain the previous value_
    }
    
    //! The server part of the implementation
    void server() {
      tcp::endpoint endpoint(tcp::v4(), port);
      tcp::acceptor acceptor(io_service, endpoint);
      while(1) {
	using namespace std;
        tcp::iostream stream;
        acceptor.accept(*stream.rdbuf());
	std::string clientname;
	getline(stream, clientname);
	cout << "server: contacted by " << clientname << endl;
        char server_averages;
	stream >> server_averages;
	
        try {
          //using namespace std;
          // cerr << stream.rdbuf()->local_endpoint() << " accepted connection from "
          //      << stream.rdbuf()->remote_endpoint() << ", " 
          //      << (server_averages ? "server averages":"client averages") << endl;
          if (server_averages) 
            perform_averaging(stream);
          else
            assist_averaging(stream);

	  cout << "server: completed " << clientname << endl;
        } catch (std::exception& e) {
	  cout << "server: aborted " << clientname << endl;
	  cout << "        " << e.what() << endl;
	} 
      }
    }

    //! The client part of the implementation
    bool client() {
      std::string hostname2;
      unsigned short port2;
      try {
	boost::tie(hostname2, port2) = (*hostfn)();
      } catch(std::exception& e) {
	return false;
      }
      if (hostname == hostname2 && port == port2)
        return true; // noop; if we did not return, we would deadlock on mutex
      
      using namespace std;
      // cerr << hostname << ":" << port << " contacting "
      //      << hostname2 << ":" << port2 << endl;

      tcp::iostream stream(hostname2,boost::lexical_cast<std::string>(port2));
      if (!stream) return false;
      //cout << "client: contacting " << hostname2 << endl;

      char server_averages = 
        std::make_pair(hostname, port) < std::make_pair(hostname2, port2);
      stream << hostname << std::endl;
      stream << server_averages;
      cout << "client: contacting " << hostname2 
	   << (server_averages ? " (server averages)" : " (client averages)")
	   << endl;

      try {
        if (server_averages)
          assist_averaging(stream);
        else
          perform_averaging(stream);
	cout << "client: completed " << hostname2 << endl;
        return true;
      } catch (std::exception& e) { 
	cout << "client: aborted " << hostname2 << endl;
	cout << "        " << e.what() << endl;
        return false; // ignore an unsuccessful attempt
      } 
      return true;
    }
    
    //! Perform one step of averaging
    bool iterate() {
      return client();
    }

  }; // class pairwise_averaging

}

#endif

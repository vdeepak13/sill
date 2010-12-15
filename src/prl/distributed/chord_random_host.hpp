#ifndef PRL_CHORD_RANDOM_HOST_HPP
#define PRL_CHORD_RANDOM_HOST_HPP

#include <prl/distributed/random_host.hpp>

// forward declarations
// (so that we do not have to include all of Chord's & sfslite's messy headers)
struct chord_node;
struct bigint;
typedef bigint chordID;

namespace prl {

  /**
   * A function object that uses Chord to return a random host-port pair.
   * Note that this object indirectly uses the POSIX random() function,
   * so it may be necesary to initialize the random sequence using srandom().
   *
   * Note: this object is not thread-safe; operator()() should not be called
   *       by multiple threads at the same time.
   */
  class chord_random_host : public random_host {

  private:
    //! A node that processes our lookup requests. Usually the local node.
    //! This variable is allocated in the constructor. 
    chord_node& contact_node;

    unsigned short client_port;
    
    //! Did the most recent lookup succeed?
    mutable bool success;

    //! The resulting random node
    //! This variable is allocated in the construtor.
    mutable chord_node& random_node;

  public:

    /**
     * Create the object with the specified node that will process the 
     * Chord requests. 
     */
    chord_random_host(const std::string& ip_address,
		      unsigned short chord_port,
		      unsigned short client_port);

    ~chord_random_host();

    std::pair<std::string, unsigned short> operator()() const;

    //! Invokes the findroute RPC (used internally)
    void findroute(const chordID& key) const;

    //! Sets the resulting random_node (used internally)
    void set_result(const chord_node& result) const;

    //! The RPC failed
    void failed() const;

  }; // class chord_random_host
  
} // namespace prl

#endif

#ifndef PRL_CHORD_LOOKUP_HPP
#define PRL_CHORD_LOOKUP_HPP

#include <map> 
#include <string>
#include <vector>

#include <prl/global.hpp>

struct chord_node;
struct bigint;
typedef bigint chordID;

namespace prl {

  class chord_lookup {
    
    //! A node that processes our lookup requests
    chord_node& contact_node;
    
    //! A map that holds the result
    std::map<chordID, chord_node>& successor;

    //! A list of IDs to be looked up
    std::vector<chordID>& ids;

    //! Number of remaining requests
    size_t nrequests;
    
  public:
    //! Constructs the object with the specified chord address and port
    chord_lookup(const std::string& ip_address, unsigned short chord_port);

    //! Destructor
    ~chord_lookup();
    
    //! Invokes the lookup RPC for the specified id
    void lookup(const chordID& key);

    //! Listens for the RPC results
    void run();
    
    //! Decreases the number of remaining requests (used internally).
    size_t set_result();
    
    //! Sets the result & decreases the number of remaining requests
    size_t set_result(const chordID& key, const chord_node& node);

    //! Returns the lookup results
    const std::map<chordID, chord_node>& result() const {
      return successor;
    }

    //! Returns the chord node associated with a key
    const chord_node& result(const chordID& key) const;

    //! Returns the address of the node associated with a key
    std::string result_address(const chordID& key) const;
    
    //! Returns true if the lookup for the given node succeeded
    bool contains(const chordID& key) const;

  }; // class chord_lookup

}


#endif

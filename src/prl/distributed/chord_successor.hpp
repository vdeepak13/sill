#ifndef PRL_CHORD_SUCCESSOR_HPP
#define PRL_CHORD_SUCCESSOR_HPP

#include <prl/global.hpp>

struct chord_node;
struct bigint;
typedef bigint chordID;

namespace prl {

  class chord_successor {

    //! A node whose successor we are trying to obtain
    chord_node& contact;

    //! The resulting successor
    chord_node& successor;

    //! Did the query succeed?
    bool success;

  public:
    //! Constructs the object with the specified chord address and port
    chord_successor(const std::string& ip_address, unsigned short chord_port);

    //! Constructs the object with the specified contact node
    chord_successor(const chord_node& contact);

    //! Destructor
    ~chord_successor();

    //! Returns our node
    const chord_node& my_node();

    //! Returns our id
    const chordID& my_id();
    
    //! Returns the current successor node
    const chord_node& successor_node();

    //! Returns the current successor id
    const chordID& successor_id();

    //! Sets the resulting successor (used internally)
    void set_result(const chord_node& node);

    //! Sets the result to failure (used internally)
    void set_result();
  };

}

#endif

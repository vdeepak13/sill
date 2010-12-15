#ifndef PRL_CHORD_PREDECESSOR_HPP
#define PRL_CHORD_PREDECESSOR_HPP

#include <prl/global.hpp>

struct chord_node;
struct bigint;
typedef bigint chordID;

namespace prl {

  class chord_predecessor {

    //! A node whose predecessor we are trying to obtain
    chord_node& contact;

    //! The resulting predecessor
    chord_node& predecessor;

    //! Did the query succeed?
    bool success;

  public:
    //! Constructs the object with the specified chord address and port
    chord_predecessor(const std::string& ip_address, unsigned short chord_port);

    //! Constructs the object with the specified contact node
    chord_predecessor(const chord_node& contact);

    //! Destructor
    ~chord_predecessor();

    //! Returns our node
    const chord_node& my_node();

    //! Returns our id
    const chordID& my_id();
    
    //! Returns the current predecessor node
    const chord_node& predecessor_node();

    //! Returns the current predecessor id
    const chordID& predecessor_id();

    //! Sets the resulting predecessor (used internally)
    void set_result(const chord_node& node);

    //! Sets the result to failure (used internally)
    void set_result();
  };

}

#endif

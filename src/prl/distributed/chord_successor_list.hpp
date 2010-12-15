#ifndef PRL_CHORD_SUCCESSOR_HPP
#define PRL_CHORD_SUCCESSOR_HPP

#include <vector>

#include <prl/global.hpp>

struct chord_node;
struct bigint;
typedef bigint chordID;

namespace prl {

  class chord_successor_list {

    //! A node whose successor we are trying to obtain
    chord_node& contact;

    //! The resulting successor
    std::vector<chord_node>& successors;

    //! Did the query succeed?
    bool success;

  public:
    //! Constructs the object with the specified chord address and port
    chord_successor_list(const std::string& ip_address,
			 unsigned short chord_port);

    //! Constructs the object with the specified contact node
    chord_successor_list(const chord_node& contact);

    //! Destructor
    ~chord_successor_list();

    //! Returns our node
    const chord_node& my_node();

    //! Returns our id
    const chordID& my_id();
    
    //! Returns the current successor node
    const std::vector<chord_node>& successor_nodes();

    //! Returns the current successor id
    std::vector<chordID> successor_ids();

    //! Sets the resulting successor (used internally)
    void set_result(const std::vector<chord_node>& node);

    //! Sets the result to failure (used internally)
    void set_result();
  };

}

#endif

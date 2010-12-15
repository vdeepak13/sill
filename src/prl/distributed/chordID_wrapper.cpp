#include <prl/distributed/chordID_wrapper.hpp>

#include <chord_types.h>
#include <id_utils.h>

namespace prl {

  chordID_wrapper::chordID_wrapper() 
    : x(*new chordID()) { }

  chordID_wrapper::chordID_wrapper(const chordID& x)
    : x(*new chordID(x)) { }

  chordID_wrapper::chordID_wrapper(const chordID_wrapper& other) 
    : x(*new chordID(other.x)) { }

  chordID_wrapper::chordID_wrapper(size_t u) 
    : x(*new chordID(make_chordID("", 0, u))) { }

  chordID_wrapper::~chordID_wrapper() {
    delete &x;
  }

  chordID_wrapper& chordID_wrapper::operator=(const chordID_wrapper& other) {
    x = other.x;
    return *this;
  }

  bool operator==(const chordID_wrapper& a, const chordID_wrapper& b) {
    return a.x == b.x;
  }

  bool operator!=(const chordID_wrapper& a, const chordID_wrapper& b) {
    return a.x != b.x;
  }

}

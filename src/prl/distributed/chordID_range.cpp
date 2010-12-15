#include <prl/distributed/chordID_range.hpp>

#include <id_utils.h>

namespace prl {

  chordID_range::chordID_range() { } 

  chordID_range::chordID_range(const chordID& a, const chordID& b)
    : a(a), b(b) { }

  bool chordID_range::contains(const chordID& x) {
    return betweenrightincl(a, b, x);
  }

}

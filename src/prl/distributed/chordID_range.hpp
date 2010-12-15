#ifndef PRL_CHORDID_RANGE_HPP
#define PRL_CHORDID_RANGE_HPP

#include <prl/distributed/chordID_wrapper.hpp>

namespace prl {

  /**
   * A half-open range (a; b] of Chord ids (with wrap-around).
   */
  struct chordID_range {
    
    chordID_wrapper a;
    chordID_wrapper b;

    chordID_range();
    chordID_range(const chordID& a, const chordID& b);
    
    bool contains(const chordID& x);

  }; // class chordID_range

}

#endif

#ifndef PRL_CHORDID_WRAPPER_HPP
#define PRL_CHORDID_WRAPPER_HPP

#include <prl/global.hpp>

class bigint;
typedef bigint chordID;

namespace prl {

  /**
   * A wrapper around the chordID type. 
   * Helps us avoid including all of Chord's & SFSlite's messy header files.
   */
  struct chordID_wrapper {
    chordID& x;

    chordID_wrapper();

    chordID_wrapper(const chordID& x);

    chordID_wrapper(const chordID_wrapper& other);
    
    explicit chordID_wrapper(size_t u);

    ~chordID_wrapper();

    chordID_wrapper& operator=(const chordID_wrapper& other);

    operator const chordID&() const { 
      return x;
    }

    operator chordID&() { 
      return x;
    }

  };

  bool operator==(const chordID_wrapper& a, const chordID_wrapper& b);

  bool operator!=(const chordID_wrapper& a, const chordID_wrapper& b);

}


#endif

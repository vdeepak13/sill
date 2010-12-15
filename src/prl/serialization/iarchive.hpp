#ifndef PRL_IARCHIVE_HPP
#define PRL_IARCHIVE_HPP

#include <iostream>
#include <cassert>
#include <string>
#include <utility>
#include <stdint.h>
#ifdef _MSC_VER
#include <itpp/base/ittypes.h> // for int32_t etc.
#endif

namespace prl {

  class universe;

  class iarchive {
  public:
    std::istream* i;
    prl::universe* u;
    size_t bytes_;

    iarchive(std::istream& is)
      :i(&is), u(NULL), bytes_() { }

    void attach_universe(prl::universe* uni) {
      u = uni;
    }

    prl::universe* universe() {
      return u;
    }

    size_t bytes() const {
      return bytes_;
    }
  };


  /** Deserializes a single character. 
      Assertion fault on failure. */
  iarchive& operator>>(iarchive& a, char& i);

  /** Deserializes a 64 bit integer. 
      Assertion fault on failure. */


  /** Deserializes a boolean. Assertion fault on failure. */
  iarchive& operator>>(iarchive& a, bool& i);

  /** Deserializes a unsigned char.
      Assertion fault on failure. */
  iarchive& operator>>(iarchive& a, unsigned char& i);

  
  iarchive& operator>>(iarchive& a, int& i);
  iarchive& operator>>(iarchive& a, long& i);
  iarchive& operator>>(iarchive& a, long long& i);
  iarchive& operator>>(iarchive& a, unsigned long& i);
  iarchive& operator>>(iarchive& a, unsigned int& i);
  iarchive& operator>>(iarchive& a, unsigned long long& i);
  iarchive& deserialize_64bit_integer(iarchive& a, int64_t& i);


  /** Deserializes a floating point number. 
      Assertion fault on failure. */
  iarchive& operator>>(iarchive& a, float& i);

  /** Serializes a double precisition floating point number. 
      Assertion fault on failure. */
  iarchive& operator>>(iarchive& a, double& i);

  /** Deserializes a generic pointer object of known length.
      This call must match the corresponding serialize call : 
      \see{serialize(std::ostream &o, const iarchive&* i,const int length)}
      The length of the object is read from the file and checked against the 
      length parameter. If they do not match, the function returns with a failure.
      Otherwise, an additional (length) bytes will be read from the file stream
      into (*i). (*i) must contain at least (length) bytes of memory. Otherwise
      there will be a buffer overflow. 
      Assertion fault on failure. */
  iarchive& deserialize(iarchive& a, void* const i, const size_t length);

  /** Loads a C string. If s is NULL, it will allocate it */
  iarchive& operator>>(iarchive& a, char*& s);

  /** Loads a string.
     Assertion fault on failure. */
  iarchive& operator>>(iarchive& a, std::string& s);


  /** Deserializes a pair
     Assertion fault on failure.  */
  template <typename T,typename U>
  iarchive& operator>>(iarchive& a, std::pair<T,U>& p){
    a >> p.first;
    a >> p.second;
    return a;
  }

  /** Catch all serializer as member of iarchive */
  template <typename T>
  inline iarchive& operator>>(iarchive& a, T& t) {
    t.load(a);
    return a;
  }
} // namespace prl

#endif //PRL_IARCHIVE_HPP

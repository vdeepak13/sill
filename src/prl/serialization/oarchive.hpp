#ifndef PRL_OARCHIVE_HPP
#define PRL_OARCHIVE_HPP

#include <iostream>
#include <cassert>
#include <string>
#include <utility>
#include <stdint.h>
#ifdef _MSC_VER
#include <itpp/base/ittypes.h> // for int32_t etc.
#endif

namespace prl {
  
  class oarchive{
   public:
    //! The associated stream
    std::ostream* o;

    //! The number of serialized bytes
    size_t bytes_;

    oarchive(std::ostream& os)
      : o(&os), bytes_() {}

    size_t bytes() const {
      return bytes_;
    }

  };

  /** Serializes a single character. 
      Assertion fault on failure. */
  oarchive& operator<<(oarchive& a, const char i);

  /** Serializes a floating point number. 
      Assertion fault on failure. */
  oarchive& operator<<(oarchive& a, const float i);

  /** Serializes a double precisition floating point number. 
      Assertion fault on failure. */
  oarchive& operator<<(oarchive& a, const double i);

  oarchive& operator<<(oarchive& a, const bool i);
  oarchive& operator<<(oarchive& a, const unsigned char i);
  
  
  /** Serializes a integer. 
      Assertion fault on failure. */
  oarchive& operator<<(oarchive& a, const int i);
  oarchive& operator<<(oarchive& a, const long i);
  oarchive& operator<<(oarchive& a, const long long i);
  oarchive& operator<<(oarchive& a, const unsigned long i);
  oarchive& operator<<(oarchive& a, const unsigned int i);
  oarchive& operator<<(oarchive& a, const unsigned long long  i);
  oarchive& serialize_64bit_integer(oarchive& a, const int64_t i);


  /** Serializes a generic pointer object. bytes from (i) to (i + length - 1) 
      inclusive will be written to the file stream. The length will also be
      Assertion fault on failure.  */
  oarchive& serialize(oarchive& a, const void* i,const size_t length);

  /** Serializes a C string 
      Assertion fault on failure. */
  oarchive& operator<<(oarchive& a, const char* s);

  /** Serializes a string. 
      Assertion fault on failure.  */
  oarchive& operator<<(oarchive  &a, const std::string& s);


  /** Serializes a pair
      Assertion fault on failure.   */
  template <typename T,typename U>
  oarchive& operator<<(oarchive& a, const std::pair<T,U>& p) {
    a << p.first;
    a << p.second;
    return a;
  }

  /** catch all operator<< as member of iarchive */
  template <typename T>
  inline oarchive& operator<<(oarchive& a, const T& t) {
    t.save(a);
    return a;
  }
} // namespace prl

#endif  //PRL_OARCHIVE_HPP

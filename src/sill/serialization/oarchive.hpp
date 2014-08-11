#ifndef SILL_OARCHIVE_HPP
#define SILL_OARCHIVE_HPP

#include <cassert>
#include <iostream>
#include <stdint.h>
#include <string>
#include <utility>

#include <boost/noncopyable.hpp>

namespace sill {

  /**
   * A class for serializing data in the native binary format.
   * The class is used in conjunction wtih operator<<. By default,
   * operator>> throws an exception if the write operation fails.
   */
  class oarchive : boost::noncopyable {
  public:
    std::ostream* o;  //!< The associated stream
    size_t bytes_;    //!< The number of serialized bytes.

    oarchive(std::ostream& os)
      : o(&os), bytes_() {}
    
    void reset(std::ostream& os) {
      o = &os;
      bytes_ = 0;
    }

    size_t bytes() const {
      return bytes_;
    }

    void check() {
      if (o->fail()) {
        throw std::runtime_error("oarchive: Stream operation failed!");
      }
    }
  };

  //! Serializes a single character. \relates oarchive
  oarchive& operator<<(oarchive& a, const char c);

  //! Serializes a single character. \relates oarchive
  oarchive& operator<<(oarchive& a, const unsigned char c);

  //! Serializes a primitive type. \relates oarchive
  oarchive& operator<<(oarchive& a, const bool b);

  //! Serializes a primitive type. \relates oarchive
  oarchive& operator<<(oarchive& a, const int x);

  //! Serializes a primitive type. \relates oarchive
  oarchive& operator<<(oarchive& a, const long x);

  //! Serializes a primitive type. \relates oarchive
  oarchive& operator<<(oarchive& a, const long long x);

  //! Serializes a primitive type. \relates oarchive
  oarchive& operator<<(oarchive& a, const unsigned long x);

  //! Serializes a primitive type. \relates oarchive
  oarchive& operator<<(oarchive& a, const unsigned int x);

  //! Serializes a primitive type. \relates oarchive
  oarchive& operator<<(oarchive& a, const unsigned long long x);

  //! Serializes a floating point number. \relates oarchive
  oarchive& operator<<(oarchive& a, const float x);

  //! Serializes a floating point number. \relates oarchive
  oarchive& operator<<(oarchive& a, const double x);

  /**
   * Serializes a generic pointer object. bytes from (i) to (i + length - 1) 
   * inclusive will be written to the archive. The length will also be
   * serialized.
   * \relates oarchive
   */
  oarchive& serialize(oarchive& a, const void* i, const size_t length);

  //! Serializes a C string. \relates oarchive
  oarchive& operator<<(oarchive& a, const char* s);

  //! Serializes a string. \relates oarchive
  oarchive& operator<<(oarchive& a, const std::string& s);

  //! Serializes a pair. \relates oarchive
  template <typename T,typename U>
  oarchive& operator<<(oarchive& a, const std::pair<T,U>& p) {
    a << p.first;
    a << p.second;
    return a;
  }

  /**
   * Catch all serializer that invokes save() member of the class T.
   * \relates oarchive
   */
  template <typename T>
  inline oarchive& operator<<(oarchive& a, const T& t) {
    t.save(a);
    return a;
  }

} // namespace sill

#endif

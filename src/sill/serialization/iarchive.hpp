#ifndef SILL_IARCHIVE_HPP
#define SILL_IARCHIVE_HPP

#include <cassert>
#include <iostream>
#include <stdint.h>
#include <string>
#include <utility>

#include <boost/noncopyable.hpp>

namespace sill {

  class universe;

  /**
   * A class for deserializing data in the native binary format.
   * The class is used in conjunction with operator>>. By default,
   * operator>> throws an exception if the read operation fails.
   */
  class iarchive : boost::noncopyable {
  public:
    std::istream* i;
    sill::universe* u;
    size_t bytes_;

    iarchive(std::istream& is)
      :i(&is), u(NULL), bytes_() { }

    void attach_universe(sill::universe* uni) {
      u = uni;
    }

    sill::universe* universe() {
      return u;
    }

    size_t bytes() const {
      return bytes_;
    }

    void check() {
      if (i->fail()) {
        throw std::runtime_error("iarchive: Stream operation failed!");
      }
    }
  };


  //! Deserializes a single character. \relates iarchive
  iarchive& operator>>(iarchive& a, char& c);

  //! Deserializaes a single character. \relates iarchive
  iarchive& operator>>(iarchive& a, unsigned char& c);

  //! Deserializaes a primitive type. \relates iarchive
  iarchive& operator>>(iarchive& a, bool& b);
 
  //! Deserializaes a primitive type. \relates iarchive
  iarchive& operator>>(iarchive& a, int& x);

  //! Deserializaes a primitive type. \relates iarchive
  iarchive& operator>>(iarchive& a, long& x);

  //! Deserializaes a primitive type. \relates iarchive
  iarchive& operator>>(iarchive& a, long long& x);

  //! Deserializaes a primitive type. \relates iarchive
  iarchive& operator>>(iarchive& a, unsigned long& x);

  //! Deserializaes a primitive type. \relates iarchive
  iarchive& operator>>(iarchive& a, unsigned int& x);

  //! Deserializaes a primitive type. \relates iarchive
  iarchive& operator>>(iarchive& a, unsigned long long& x);

  //! Deserializes a floating point number. \relates iarchive
  iarchive& operator>>(iarchive& a, float& x);

  //! Deserializes a floating point number. \relates iarchive
  iarchive& operator>>(iarchive& a, double& x);

  /** 
   * Deserializes a generic pointer object of known length.
   * This call must match the corresponding serialize call: 
   * \see{serialize(std::ostream &o, const iarchive&* i,const int length)}
   * The length of the object is read from the file and checked against the 
   * length parameter. If they do not match, the function throws assertion.
   * Otherwise, an additional (length) bytes will be read from the file stream
   * into (*i). (*i) must contain at least (length) bytes of memory. Otherwise
   * there will be a buffer overflow.
   * \relates iarchive
   */
  iarchive& deserialize(iarchive& a, void* const x, const size_t length);

  //! Deserializes a C string. If s is NULL, it will allocate it.
  //! \relates iarchive
  iarchive& operator>>(iarchive& a, char*& s);

  //! Loads a string. \relates iarchive
  iarchive& operator>>(iarchive& a, std::string& s);

  //! Deserializes a pair. \relates iarchive
  template <typename T,typename U>
  inline iarchive& operator>>(iarchive& a, std::pair<T,U>& p){
    a >> p.first;
    a >> p.second;
    return a;
  }

  /**
   * Catch all deserializer that invokes a load() member of the class T.
   * \relates iarchive
   */
  template <typename T>
  inline iarchive& operator>>(iarchive& a, T& t) {
    t.load(a);
    return a;
  }

} // namespace sill

#endif

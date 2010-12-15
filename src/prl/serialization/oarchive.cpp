#include <iostream>
#include <cstring>
#include <stdexcept>

#include <prl/serialization/oarchive.hpp>

/***************************************************************************
 *                        Basic Serializers                                * 
 ***************************************************************************/
// generate the operator<< call for a whole bunch of integer types
#define GENCASTSERIALIZE(typesrc,typedest)              \
  oarchive& operator<<(oarchive& a, const typesrc i) {  \
    operator<<(a, static_cast<const typedest>(i));      \
    return a;                                           \
  }
  
#define GENINTSERIALIZE(typesrc)                              \
  oarchive& operator<<(oarchive& a, const typesrc i) {        \
    serialize_64bit_integer(a, static_cast<int64_t>(i));      \
    return a;                                                 \
  }

namespace {

  inline void check(std::ostream* out) {
    if (out->fail()) {
      throw std::runtime_error("oarchive: Stream operation failed!");
    }
  }

}

namespace prl {

  oarchive& operator<<(oarchive& a, const char i) {
    a.o->put(i);
    a.bytes_++;
    check(a.o);
    return a;
  }

  oarchive& serialize_64bit_integer(oarchive& a, const int64_t i) {
    a.o->write(reinterpret_cast<const char*>(&i), sizeof(int64_t));
    a.bytes_ += sizeof(int64_t);
    check(a.o);
    return a;
  }

  oarchive& operator<<(oarchive& a, const float i) {
    a.o->write(reinterpret_cast<const char*>(&i), sizeof(float));
    a.bytes_ += sizeof(float);
    check(a.o);
    return a;
  }

  oarchive& operator<<(oarchive& a, const double i) {
    a.o->write(reinterpret_cast<const char*>(&i), sizeof(double));
    a.bytes_ += sizeof(double);
    check(a.o);
    return a;
  }

  GENCASTSERIALIZE(bool, char);
  GENCASTSERIALIZE(unsigned char, char);
  GENINTSERIALIZE(int);
  GENINTSERIALIZE(long);
  GENINTSERIALIZE(long long);
  GENINTSERIALIZE(unsigned long);
  GENINTSERIALIZE(unsigned int);
  GENINTSERIALIZE(unsigned long long);
  //GENCASTSERIALIZE(size_t, int64_t);

  oarchive& serialize(oarchive& a, const void* i, const size_t length) {
    // save the length
    operator<<(a,length);
    a.o->write(reinterpret_cast<const char*>(i), length);
    a.bytes_ += length;
    check(a.o);
    return a;
  }

  oarchive& operator<<(oarchive& a, const char* s) {
    // save the length
    size_t length = strlen(s);
    operator<<(a,length);
    a.o->write(reinterpret_cast<const char*>(s), length);
    a.bytes_ += length;
    check(a.o);
    return a;
  }

  oarchive& operator<<(oarchive& a, const std::string& s){
    // if we can't serialize the length, return immediately
    size_t length = s.length();
    a << length;
    a.o->write(reinterpret_cast<const char*>(s.c_str()), length);
    a.bytes_ += length;
    check(a.o);
    return a;
  }

} // namespace prl

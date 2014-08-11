#include <sill/serialization/oarchive.hpp>

#include <cstring>
#include <iostream>
#include <stdexcept>

#define SILL_CHAR_SERIALIZE(src_type)                       \
  oarchive& operator<<(oarchive& a, const src_type c) {     \
    serialize_character(a, static_cast<const src_type>(c)); \
    return a;                                               \
  }

#define SILL_INT64_SERIALIZE(src_type)                     \
  oarchive& operator<<(oarchive& a, const src_type x) {     \
    serialize_64bit_integer(a, static_cast<int64_t>(x));    \
    return a;                                               \
  }

namespace sill {

  void serialize_character(oarchive& a, const char c) {
    a.o->put(c);
    a.bytes_++;
    a.check();
  }

  void serialize_64bit_integer(oarchive& a, const int64_t x) {
    a.o->write(reinterpret_cast<const char*>(&x), sizeof(int64_t));
    a.bytes_ += sizeof(int64_t);
    a.check();
  }

  SILL_CHAR_SERIALIZE(char);
  SILL_CHAR_SERIALIZE(unsigned char);
  SILL_CHAR_SERIALIZE(bool);

  SILL_INT64_SERIALIZE(int);
  SILL_INT64_SERIALIZE(long);
  SILL_INT64_SERIALIZE(long long);
  SILL_INT64_SERIALIZE(unsigned long);
  SILL_INT64_SERIALIZE(unsigned int);
  SILL_INT64_SERIALIZE(unsigned long long);

  oarchive& operator<<(oarchive& a, const float x) {
    a.o->write(reinterpret_cast<const char*>(&x), sizeof(float));
    a.bytes_ += sizeof(float);
    a.check();
    return a;
  }

  oarchive& operator<<(oarchive& a, const double x) {
    a.o->write(reinterpret_cast<const char*>(&x), sizeof(double));
    a.bytes_ += sizeof(double);
    a.check();
    return a;
  }
  oarchive& serialize(oarchive& a, const void* i, const size_t length) {
    a << length;
    a.o->write(reinterpret_cast<const char*>(i), length);
    a.bytes_ += length;
    a.check();
    return a;
  }

  oarchive& operator<<(oarchive& a, const char* s) {
    size_t length = strlen(s);
    a << length;
    a.o->write(reinterpret_cast<const char*>(s), length);
    a.bytes_ += length;
    a.check();
    return a;
  }

  oarchive& operator<<(oarchive& a, const std::string& s) {
    size_t length = s.length();
    a << length;
    a.o->write(reinterpret_cast<const char*>(s.c_str()), length);
    a.bytes_ += length;
    a.check();
    return a;
  }

} // namespace sill

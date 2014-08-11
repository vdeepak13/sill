#include <sill/serialization/iarchive.hpp>

#include <iostream>
#include <stdexcept>

#define SILL_CHAR_DESERIALIZE(dest_type)            \
  iarchive& operator>>(iarchive& a, dest_type& x) { \
    char c;                                         \
    deserialize_character(a, c);                    \
    x = static_cast<dest_type>(c);                  \
    return a;                                       \
  }

#define SILL_INT64_DESERIALIZE(dest_type)           \
  iarchive& operator>>(iarchive& a, dest_type& x) { \
    int64_t y;                                      \
    deserialize_64bit_integer(a, y);                \
    x = static_cast<dest_type>(y);                  \
    return a;                                       \
  }
  
namespace sill {

  iarchive& deserialize_character(iarchive& a, char& c) {
    a.i->get(c);
    a.bytes_++;
    a.check();
    return a;
  }

  iarchive& deserialize_64bit_integer(iarchive& a, int64_t& x) {
    a.i->read(reinterpret_cast<char*>(&x), sizeof(int64_t));
    a.bytes_ += sizeof(int64_t);
    a.check();
    return a;
  }


  SILL_CHAR_DESERIALIZE(bool)
  SILL_CHAR_DESERIALIZE(char)
  SILL_CHAR_DESERIALIZE(unsigned char);

  SILL_INT64_DESERIALIZE(int);
  SILL_INT64_DESERIALIZE(long);
  SILL_INT64_DESERIALIZE(long long);
  SILL_INT64_DESERIALIZE(unsigned long);
  SILL_INT64_DESERIALIZE(unsigned int);
  SILL_INT64_DESERIALIZE(unsigned long long);

  iarchive& operator>>(iarchive& a, float& x) {
    a.i->read(reinterpret_cast<char*>(&x), sizeof(float));
    a.bytes_ += sizeof(float);
    a.check();
    return a;
  }

  iarchive& operator>>(iarchive& a, double& x) {
    a.i->read(reinterpret_cast<char*>(&x), sizeof(double));
    a.bytes_ += sizeof(double);
    a.check();
    return a;
  }

  iarchive& deserialize(iarchive& a, void* const i, const size_t length) {
    // deserialize the length and check if lengths match
    size_t length2;
    a >> length2;
    assert(length == length2);

    // deserialize the rest
    a.i->read(reinterpret_cast<char*>(i), length);
    a.bytes_ += length;
    a.check();
    return a;
  }

  iarchive& operator>>(iarchive& a, char*& s) {
    // deserialize the length
    size_t length;
    a >> length;
    if (s == NULL) {
      s = new char[length+1];
    }

    // deserialize the rest
    a.i->read(reinterpret_cast<char*>(s), length);
    s[length] = 0;
    a.bytes_ += length;
    a.check();
    return a;
  }

  iarchive& operator>>(iarchive& a, std::string& s) {
    size_t length;
    a >> length;

    s.resize(length);
    a.i->read(const_cast<char*>(s.c_str()), length);
    a.bytes_ += length;
    a.check();
    return a;
  }

} // namespace sill

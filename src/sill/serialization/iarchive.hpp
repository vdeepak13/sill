#ifndef SILL_IARCHIVE_HPP
#define SILL_IARCHIVE_HPP

#include <cassert>
#include <cstdint>
#include <iostream>

#include <boost/noncopyable.hpp>

#define SILL_DESERIALIZE_CHAR(dest_type)            \
  iarchive& operator>>(dest_type& x) {              \
    x = static_cast<dest_type>(deserialize_char()); \
    return *this;                                   \
  }

#define SILL_DESERIALIZE_INT(dest_type)             \
  iarchive& operator>>(dest_type& x) {              \
    x = static_cast<dest_type>(deserialize_int());  \
    return *this;                                   \
  }

namespace sill {

  // Forward declaration
  class universe;

  /**
   * A class for deserializing data in the native binary format.
   * The class is used in conjunction with operator>>. By default,
   * operator>> throws an exception if the read operation fails.
   */
  class iarchive : boost::noncopyable {
  public:
    //! Constructs an input archive with the given stream.
    iarchive(std::istream& in)
      :in_(&in), u_(NULL), bytes_(0) { }

    //! Sets the universe associated with the archive.
    void universe(sill::universe* u) {
      u_ = u;
    }

    //! Returns the universe associated with the archive.
    sill::universe* universe() {
      return u_;
    }

    //! Returns the number of bytes read.
    size_t bytes() const {
      return bytes_;
    }

    //! Throws an exception if the input stream indicates failure.
    void check() {
      if (in_->fail()) {
        throw std::runtime_error("iarchive: Stream operation failed!");
      }
    }

    //! Deserializes a single character from one byte.
    char deserialize_char() {
      char c;
      in_->get(c);
      ++bytes_;
      check();
      return c;
    }

    //! Deserializes a 64-bit integer.
    int64_t deserialize_int() {
      int64_t x;
      in_->read(reinterpret_cast<char*>(&x), sizeof(int64_t));
      bytes_ += sizeof(int64_t);
      check();
      return x;
    }

    //! Deserializes a raw buffer with length bytes.
    void deserialize_buf(void* const buf, const size_t length) {
      if (length == 0) { return; }
      in_->read(reinterpret_cast<char*>(buf), length);
      bytes_ += length;
      check();
    }
    
    //! Deserializes a range elements of type T into an output iterator.
    template <typename T, typename OutputIterator>
    OutputIterator deserialize_range(OutputIterator it) {
      size_t length = deserialize_int();
      for (size_t i = 0; i < length; ++i) {
        T value;
        *this >> value;
        *it = value;
        ++it;
      }
      return it;
    }

    SILL_DESERIALIZE_CHAR(bool)
    SILL_DESERIALIZE_CHAR(char)
    SILL_DESERIALIZE_CHAR(unsigned char);
    
    SILL_DESERIALIZE_INT(int);
    SILL_DESERIALIZE_INT(long);
    SILL_DESERIALIZE_INT(long long);
    SILL_DESERIALIZE_INT(unsigned long);
    SILL_DESERIALIZE_INT(unsigned int);
    SILL_DESERIALIZE_INT(unsigned long long);

    iarchive& operator>>(float& x) {
      in_->read(reinterpret_cast<char*>(&x), sizeof(float));
      bytes_ += sizeof(float);
      check();
      return *this;
    }

    iarchive& operator>>(double& x) {
      in_->read(reinterpret_cast<char*>(&x), sizeof(double));
      bytes_ += sizeof(double);
      check();
      return *this;
    }

  private:
    std::istream* in_;  //!< The stream from which we read data.
    sill::universe* u_; //!< The attached universe.
    size_t bytes_;      //!< The number of bytes read.
  };

  /**
   * Catch all deserializer that invokes a load() member of the class T.
   * \relates iarchive
   */
  template <typename T>
  iarchive& operator>>(iarchive& a, T& t) {
    t.load(a);
    return a;
  }

} // namespace sill

#endif

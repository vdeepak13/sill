
#ifndef PRL_PTR_VECTOR_HPP
#define PRL_PTR_VECTOR_HPP

#include <fstream>
#include <string>
#include <sstream>
#include <vector>

#include <boost/iterator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

#include <prl/assignment.hpp>
#include <prl/factor/table_factor.hpp>
#include <prl/learning/dataset/concepts.hpp>
#include <prl/learning/dataset/dataset.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  /**
   * This is a class which is about the same as std::vector<T> but allows
   * you to get a pointer to a value in the vector, rather than only a
   * reference.
   * \todo Move this to datastructures.
   */
  template <typename T>
  class ptr_vector {

  public:
    typedef T value_type;

  private:
    static const size_t INIT_CAP = 4;
    //! Array of values
    T* values;
    //! Capacity of values
    size_t cap;
    //! Number of values currently in vector
    size_t n;

    //! Grow capacity to max(2 * cap, newcap)
    void grow(size_t newcap) {
      size_t m = 2 * cap;
      if (m < newcap)
        m = newcap;
      assert(m > cap); // in case reach size_t's max
      T* newvalues = new T[m];
      for (size_t i = 0; i < n; ++i)
        newvalues[i] = values[i];
      delete[] values;
      values = newvalues;
      cap = m;
    }

    ///////////////////// PUBLIC MEMBERS: CONSTRUCTORS //////////////////////

  public:
    //! Default constructor
    ptr_vector() : values(new T[INIT_CAP]), cap(INIT_CAP), n(0) { }
    //! Copy constructor
    ptr_vector( const ptr_vector& c )
      : values(new T[c.cap]), cap(c.cap), n(c.n) {
      for (size_t i = 0; i < n; ++i)
        values[i] = c.values[i];
    }
    //! Constructor which allocates space for num values, with default value val
    ptr_vector( size_t num, const T& val = T() )
      : values(new T[num]), cap(num), n(num) {
      for (size_t i = 0; i < n; ++i)
        values[i] = val;
    }
//    ptr_vector( input_iterator start, input_iterator end ) { }
    ~ptr_vector() {
      delete[] values;
    }

    ///////////////////// PUBLIC MEMBERS: OPERATORS //////////////////////

    //! NOTE: Like std::vector<T>, this does not check index bounds!
    T& operator[]( size_t index ) { return values[index]; }
    //! NOTE: Like std::vector<T>, this does not check index bounds!
    const T& operator[]( size_t index ) const { return values[index]; }
    ptr_vector& operator=(const ptr_vector& c) {
      delete[] values;
      values = new T[c.cap];
      cap = c.cap;
      n = c.n;
      for (size_t i = 0; i < n; ++i)
        values[i] = c.values[i];
      return *this;
    }
    bool operator==(const ptr_vector& c) {
      if (n != c.n)
        return false;
      for (size_t i = 0; i < n; ++i)
        if (values[i] != c.values[i])
          return false;
      return true;
    }
    bool operator!=(const ptr_vector& c) {
      return !operator==(c);
    }
//    bool operator<(const ptr_vector& c) { }
//    bool operator>(const ptr_vector& c) { }
//    bool operator<=(const ptr_vector& c) { }
//    bool operator>=(const ptr_vector& c) { }

    ///////////////////// PUBLIC MEMBERS: FUNCTIONS //////////////////////

    T* get_ptr(size_t index) {
      assert(index < n);
      return &(values[index]);
    }

    void resize( size_t num, const T& val = T() ) {
      if (num < n) {
        n = num;
      } else if (num > n) {
        if (cap < num)
          grow(num);
        for (size_t i = n; i < num; ++i)
          values[i] = val;
        n = num;
      }
    }
    size_t size() const { return n; }
  };

} // namespace prl

#include <prl/macros_undef.hpp>

#endif

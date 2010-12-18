
#ifndef SILL_DISCRIMINATIVE_FREE_FUNCTIONS_HPP
#define SILL_DISCRIMINATIVE_FREE_FUNCTIONS_HPP

#include <sill/learning/discriminative/concepts.hpp>
#include <sill/math/matrix.hpp>
#include <sill/math/vector.hpp>
#include <sill/stl_io.hpp>

#include <sill/macros_def.hpp>

namespace sill {

#ifndef SWIG
  //! Absolute value
  //! @todo Put this somewhere else.
  template <typename T>
  inline T absval(T val) { return (val >= 0 ? val : -val); }
#else
  inline double absval(double val) { return fabs(val); }
#endif

  //! Return +1 if label > 0, else -1
  inline double binary_label(double label) { return (label > 0 ? 1 : -1); }

  //! Read in a vector of values [val1,val2,...], ignoring an initial space
  //! if necessary.
  //! \todo Can we overload operator<< for this?
  template <typename Char, typename Traits, typename T>
  static void read_vec
  (std::basic_istream<Char,Traits>& in, vector<T>& v) {
    char c;
    T val;
    v.resize(0);
    in.get(c);
    if (c == ' ')
      in.get(c);
    assert(c == '[');
    if (in.peek() != ']') {
      do {
        if (!(in >> val))
          assert(false);
        v.insert(v.size(),val);
        if (in.peek() == ',')
          in.ignore(1);
      } while (in.peek() != ']');
    }
    in.ignore(1);
  }

  //! Sets the pre-allocated vals to be the values for variables vars
  //! in the given record.
  //! This does not check the size of vals.
  void record2vector(vec& vals, const vector_var_vector& vars,
                     const vector_record& r);

  //! Sets the pre-allocated vals to be the values for variables vars
  //! in the given assignment.
  //! This does not check the size of vals.
  void assignment2vector(vec& vals, const vector_var_vector& vars,
                         const vector_assignment& r);

} // namespace sill


#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_DISCRIMINATIVE_FREE_FUNCTIONS_HPP

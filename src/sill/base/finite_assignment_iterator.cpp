#include <sill/base/finite_assignment_iterator.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  finite_assignment_iterator::
  finite_assignment_iterator(const forward_range<finite_variable*>& vars)
    : var_vec(boost::begin(vars), boost::end(vars)), done(false) {
    foreach(finite_variable* var, var_vec) {
      a[var] = 0;
    }
  }

  finite_assignment_iterator& 
  finite_assignment_iterator::operator++() {
    foreach(finite_variable* var, var_vec) {
      size_t value = a[var] + 1;
      if (value >= var->size()) {
        a[var] = 0;
      } else {
        a[var] = value;
        return *this;
      }
    }
    done = true;
    return *this;
  }

  finite_assignment_iterator 
  finite_assignment_iterator::operator++(int) {
    finite_assignment_iterator tmp = *this;
    ++(*this);
    return tmp;
  }

  bool finite_assignment_iterator::
  operator==(const finite_assignment_iterator& it) const {
    if (done)
      return it.done;
    else
      return !it.done && (a == *it);
    // Old buggy code - does not correctly handle the empty assignment case
    // return (done && it.done) || equal(a, *it);
  }

  finite_assignment_range assignments(const finite_domain& vars) {
    return std::make_pair(finite_assignment_iterator(vars),
                          finite_assignment_iterator());
  }

}

#include <sill/macros_undef.hpp>

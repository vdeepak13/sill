#ifndef SILL_FINITE_ASSIGNMENT_HPP
#define SILL_FINITE_ASSIGNMENT_HPP

#include <boost/random/uniform_int.hpp>

#include <sill/base/finite_variable.hpp>
#include <sill/base/stl_util.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! \addtogroup base_types
  //! @{

  //! An assignment to a set of finite variables
  typedef std::map<finite_variable*, size_t> finite_assignment;

  //! Returns a new assignment for the given variable-value pair
  inline finite_assignment make_assignment(finite_variable* var, size_t value) {
    finite_assignment a;
    a.insert(std::make_pair(var, value));
    return a;
  }

  //! Returns a uniformly random to the given set of finite variables
  //! \param Engine a random number generator
  template<typename Engine>
  finite_assignment 
  random_assignment(const finite_domain& domain, Engine& rand) {
    boost::uniform_int<int> unif_int;
    finite_assignment a;
    foreach(finite_variable* v, domain) {
      a[v] = unif_int(rand, v->size());
    }
    return a;
  }


  //! Returns the number of terms for which both finite_assignments agree on
  inline size_t assignment_agreement(const finite_assignment &fa1, 
                                     const finite_assignment &fa2) {
    finite_domain allvars = set_intersect(keys(fa1), keys(fa2));
    size_t count = 0;
    foreach(finite_variable *v, allvars) {
      count += (safe_get(fa1, v) == safe_get(fa2, v));
    }
    return count;
  }


  //! @}

}

#include <sill/macros_undef.hpp>

#endif

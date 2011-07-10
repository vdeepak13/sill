#ifndef SILL_RANDOM_ASSIGNMENT_HPP
#define SILL_RANDOM_ASSIGNMENT_HPP

#include <boost/random/uniform_int.hpp>

#include <sill/factor/moment_gaussian.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! Returns a uniformly random to the given set of finite variables
  //! \param Engine a random number generator
  template<typename Engine>
  finite_assignment 
  random_assignment(const finite_domain& vars, Engine& rng) {
    boost::uniform_int<int> unif_int;
    finite_assignment a;
    foreach(finite_variable* v, vars) {
      a[v] = unif_int(rng, v->size());
    }
    return a;
  }

  //! Returns a sample from a zero-mean, unit-variance Gaussian.
  //! \param Engine a random number generator
  template<typename Engine>
  vector_assignment 
  random_assignment(const vector_domain& vars, Engine& rng) {
    vector_assignment a;
    if (vars.size() == 0)
      return a;
    vector_variable* u = *(vars.begin());
    moment_gaussian mg(make_vector(u), zeros<vec>(1), mat_1x1(1.), 1);
    foreach(vector_variable* v, vars) {
      a[v] = mg.sample(rng)[u];
    }
    return a;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_RANDOM_ASSIGNMENT_HPP

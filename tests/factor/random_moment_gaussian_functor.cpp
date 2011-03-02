
/**
 * \file random_moment_gaussian_functor.cpp
 *       Test of random_moment_gaussian_functor.
 */

#include <sill/base/universe.hpp>
#include <sill/factor/random_gaussian_functor.hpp>

#include <sill/macros_def.hpp>

using namespace sill;

int main(int argc, char** argv) {

  unsigned random_seed = time(NULL);

  boost::mt11213b rng(random_seed);
  boost::uniform_int<int> unif_int(0, std::numeric_limits<int>::max());
  universe u;
  vector_variable* Y = u.new_vector_variable(1);
  vector_variable* X = u.new_vector_variable(1);

  random_gaussian_functor<moment_gaussian> rmgf(unif_int(rng));

  std::cout << "Test: random_moment_gaussian_functor\n"
            << "---------------------------------------------" << std::endl;
  moment_gaussian P_Y(rmgf.generate_marginal(Y));
  std::cout << "Generated moment_gaussian P(Y):\n"
            << P_Y << std::endl;
  moment_gaussian P_YX(rmgf.generate_marginal(make_domain(Y,X)));
  std::cout << "Generated moment_gaussian P(Y,X):\n"
            << P_YX << std::endl;
  moment_gaussian P_Y_given_X(rmgf.generate_conditional(Y, X));
  std::cout << "Generated moment_gaussian P(Y|X):\n"
            << P_Y_given_X << std::endl;

} // main

#include <sill/macros_undef.hpp>


/**
 * \file random_mixture_functor.cpp
 *       Test of random_mixture_functor with random_moment_gaussian_functor.
 */

#include <sill/base/universe.hpp>
#include <sill/factor/random/random_mixture_functor.hpp>
#include <sill/factor/random/random_gaussian_factor_functor.hpp>

#include <sill/macros_def.hpp>

using namespace sill;

int main(int argc, char** argv) {

  unsigned random_seed = time(NULL);

  boost::mt11213b rng(random_seed);
  boost::uniform_int<int> unif_int(0, std::numeric_limits<int>::max());
  universe u;
  vector_variable* Y = u.new_vector_variable(1);
  vector_variable* X = u.new_vector_variable(1);

  random_gaussian_factor_functor<moment_gaussian> rmgf(unif_int(rng));
  random_mixture_functor<moment_gaussian> rmf(rmgf);

  std::cout << "Test: random_mixture_functor with moment_gaussian factors\n"
            << "-----------------------------------------------------------"
            << std::endl;
  mixture<moment_gaussian> P_Y(rmf.generate_marginal(Y));
  std::cout << "Generated mixture P(Y):\n"
            << P_Y << std::endl;
  mixture<moment_gaussian> P_YX(rmf.generate_marginal(make_domain(Y,X)));
  std::cout << "Generated mixture P(Y,X):\n"
            << P_YX << std::endl;
  mixture<moment_gaussian> P_Y_given_X(rmf.generate_conditional(Y, X));
  std::cout << "Generated mixture P(Y|X):\n"
            << P_Y_given_X << std::endl;

} // main

#include <sill/macros_undef.hpp>

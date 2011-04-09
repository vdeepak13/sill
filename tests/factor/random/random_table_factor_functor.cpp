
/**
 * \file random_table_factor_functor.cpp
 *       Test of random_table_factor_functor.
 */

#include <sill/base/universe.hpp>
#include <sill/factor/random/random_table_factor_functor.hpp>

#include <sill/macros_def.hpp>

using namespace sill;

int main(int argc, char** argv) {

  unsigned random_seed = time(NULL);

  boost::mt11213b rng(random_seed);
  boost::uniform_int<int> unif_int(0, std::numeric_limits<int>::max());
  universe u;
  finite_variable* Y = u.new_finite_variable(2);
  finite_variable* X = u.new_finite_variable(2);

  random_table_factor_functor rtff(unif_int(rng));

  std::cout << "Test: random_table_factor_functor\n"
            << "---------------------------------------------" << std::endl;
  table_factor P_Y(rtff.generate_marginal(Y));
  std::cout << "Generated table_factor P(Y):\n"
            << P_Y << std::endl;
  table_factor P_YX(rtff.generate_marginal(make_domain(Y,X)));
  std::cout << "Generated table_factor P(Y,X):\n"
            << P_YX << std::endl;
  table_factor P_Y_given_X(rtff.generate_conditional(Y, X));
  std::cout << "Generated table_factor P(Y|X):\n"
            << P_Y_given_X << std::endl;

} // main

#include <sill/macros_undef.hpp>

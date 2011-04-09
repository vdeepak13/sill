
/**
 * \file random_alternating_factor_functor.cpp
 *       Test of random_alternating_factor_functor.
 */

#include <sill/base/universe.hpp>
#include <sill/factor/random/random_alternating_factor_functor.hpp>
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

  random_table_factor_functor rtff1(unif_int(rng));
  rtff1.factor_choice = random_table_factor_functor::RANDOM_RANGE;
  random_table_factor_functor rtff2(unif_int(rng));
  rtff2.factor_choice = random_table_factor_functor::ASSOCIATIVE;

  random_alternating_factor_functor<random_table_factor_functor>
    raff(rtff1,rtff2);
  raff.alternation_period = 2;

  std::cout << "Test: random_alternating_factor_functor"
            << " with random_table_factor_functor\n"
            << " (with alternation_period = 2)\n"
            << "---------------------------------------------" << std::endl;
  std::cout << "Generated alternating_factor P(Y,X):\n"
            << raff.generate_marginal(make_domain(Y,X)) << std::endl;
  std::cout << "Generated alternating_factor P(Y,X):\n"
            << raff.generate_marginal(make_domain(Y,X)) << std::endl;
  std::cout << "Generated alternating_factor P(Y,X):\n"
            << raff.generate_marginal(make_domain(Y,X)) << std::endl;
  std::cout << "Generated alternating_factor P(Y,X):\n"
            << raff.generate_marginal(make_domain(Y,X)) << std::endl;

} // main

#include <sill/macros_undef.hpp>

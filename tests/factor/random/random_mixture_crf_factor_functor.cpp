
/**
 * \file random_mixture_crf_factor_functor.cpp
 *       Test of random_mixture_crf_factor_functor with
 *       random_gaussian_crf_factor_functor.
 */

#include <sill/base/universe.hpp>
#include <sill/factor/random/random_mixture_crf_factor_functor.hpp>
#include <sill/factor/random/random_gaussian_crf_factor_functor.hpp>
#include <sill/factor/random/random_gaussian_functor.hpp>

#include <sill/macros_def.hpp>

using namespace sill;

int main(int argc, char** argv) {

  unsigned random_seed = time(NULL);

  size_t k = 3;

  boost::mt11213b rng(random_seed);
  boost::uniform_int<int> unif_int(0, std::numeric_limits<int>::max());
  universe u;
  vector_variable* Y = u.new_vector_variable(1);
  vector_variable* X = u.new_vector_variable(1);

  random_gaussian_functor<moment_gaussian> rgf(unif_int(rng));
  random_gaussian_crf_factor_functor rgcff(rgf);
  random_mixture_crf_factor_functor<gaussian_crf_factor> rmcff(k, rgcff);

  std::cout << "Test: random_mixture_crf_factor_functor with"
            << " gaussian_crf_factors\n"
            << "-----------------------------------------------------------"
            << std::endl;
  mixture_crf_factor<gaussian_crf_factor> P_Y(rmcff.generate_marginal(Y));
  std::cout << "Generated mixture P(Y):\n"
            << P_Y << std::endl;
  mixture_crf_factor<gaussian_crf_factor>
    P_YX(rmcff.generate_marginal(make_domain(Y,X)));
  std::cout << "Generated mixture P(Y,X):\n"
            << P_YX << std::endl;
  mixture_crf_factor<gaussian_crf_factor>
    P_Y_given_X(rmcff.generate_conditional(Y, X));
  std::cout << "Generated mixture P(Y|X):\n"
            << P_Y_given_X << std::endl;

} // main

#include <sill/macros_undef.hpp>

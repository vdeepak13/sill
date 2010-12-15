#ifndef PRL_MATH_FREE_FUNCTIONS_HPP
#define PRL_MATH_FREE_FUNCTIONS_HPP

#include <cmath>
#include <vector> // for use in randperm

#include <boost/random/uniform_real.hpp>

namespace prl {

  //! \addtogroup math_free_functions
  //! @{

  //! Square a value.
  template <typename T> 
  T sqr(const T& value) { return value*value; }

  //! Round a value to the nearest integer.
  //! Note *.5 is rounded to *, but -*.5 is rounded to -(*+1).
  template <typename T>
  T round(T value) {
    return ceil(value - .5);
  }

  //! Returns a random permutation of 0,...,n-1.
  //! If 0 < k < n, then this only returns the first k elements of the
  //! permutation.
  //! \todo This function could possibly be sped up for this latter case.
  template <typename RandomNumberGenerator>
  std::vector<size_t> randperm(size_t n, RandomNumberGenerator& rng,
                               size_t k = 0) {
    boost::uniform_real<double> uniform_prob(0,1);
    std::vector<size_t> perm(n,0);
    for (size_t i = 0; i < n; ++i)
      perm[i] = i;
    if (k == 0)
      k = n;
    if (k > n) {
      std::cerr << "Error: randperm must be called with k <= n." << std::endl;
      assert(false);
      return std::vector<size_t>();
    }
    // Stano: changed the loop bound k-1 to k, in order to accomodate k < n
    for (size_t i = 0; i < k; ++i) {
      size_t j = (size_t)(floor(uniform_prob(rng) * (n-i))) + i;
      j = (j == n ? n-1 : j);
      size_t tmp = perm[i];
      perm[i] = perm[j];
      perm[j] = tmp;
    }
    if (k < n)
      perm.resize(k);
    return perm;
  }

  //! @}

} // namespace prl

#endif

#ifndef SILL_MATH_FREE_FUNCTIONS_HPP
#define SILL_MATH_FREE_FUNCTIONS_HPP

#include <algorithm> // change this to <utility> for C++11
#include <cmath>
#include <vector>

#include <boost/random/uniform_real.hpp>
#include <boost/random/uniform_int.hpp>

namespace sill {

  //! \addtogroup math_free_functions
  //! @{

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

  /**
   * Randomly permutes the given values.
   */
  template <typename T, typename RandomNumberGenerator>
  void permute(std::vector<T>& values, RandomNumberGenerator& rng) {
    for (size_t i = 0; i < values.size(); ++i) {
      boost::uniform_int<size_t> uniform(i, values.size() - 1);
      size_t j = uniform(rng);
      std::swap(values[i], values[j]);
    }
  }

  //! @}

} // namespace sill

#endif

#ifndef PRL_GEOMETRY_ALIGNMENT_HPP
#define PRL_GEOMETRY_ALIGNMENT_HPP

#include <stdexcept>

#include <prl/global.hpp>
#include <prl/math/mat.hpp>
#include <prl/math/vec.hpp>
#include <prl/math/linear_algebra.hpp>

// ?

#include <prl/macros_def.hpp>

namespace prl { 

  /**
   * Computes the optimal rigid rotation given a covariance matrix.
   * @return the optimal rotation matrix
   * \ingroup geometry
   */
  mat ralign(const mat& cov_xy);
  
  /**
   * Computes the rigid alignment given a sequence of corresponding points.
   * @return a pair (rotation, translation)
   *
   * \ingroup geometry
   */
  std::pair<mat, vec> ralign(const forward_range<const vec&>& src,
                             const forward_range<const vec&>& dest);
  
} // namespace prl

#include <prl/macros_undef.hpp>

#endif

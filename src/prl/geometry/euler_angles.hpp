#ifndef PRL_EULER_ANGLES_HPP
#define PRL_EULER_ANGLES_HPP

#include <prl/math/matrix.hpp>
#include <prl/math/linear_algebra.hpp>

namespace prl {

  //! \addtogroup geometry
  //! @{

  mat rotation_x(double theta) {
    mat a = identity(3);
    a(1,1) = cos(theta); a(1,2) = -sin(theta);
    a(2,1) = sin(theta); a(2,2) = cos(theta);
    return a;
  }

  mat rotation_y(double theta) {
    mat a = identity(3);
    a(2,2) = cos(theta); a(2,0) = -sin(theta);
    a(0,2) = sin(theta); a(0,0) = cos(theta);
    return a;
  }

  mat rotation_z(double theta) {
    mat a = identity(3);
    a(0,0) = cos(theta); a(0,1) = -sin(theta);
    a(1,0) = sin(theta); a(1,1) = cos(theta);
    return a;
  }

  //! @}

} // namespace prl

#endif

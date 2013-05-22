
#ifndef _SILL_GEOMETRY_HPP_
#define _SILL_GEOMETRY_HPP_

#include <cmath>

#include <sill/math/constants.hpp>
#include <sill/math/linear_algebra/armadillo.hpp>

namespace sill {

  //! Convert degrees to radians.
  inline double degree2radian(double angle) {
    return angle * (pi() / 180.);
  }

  //! Convert radians to degrees.
  inline double radian2degree(double angle) {
    return angle * (180. / pi());
  }

  /**
   * Convert spherical coordinates to Cartesian coordinates.
   * All angles are in radians.
   * @param x  [theta, phi, radius], where theta in [0,pi], phi in [-pi,pi]
   */
  inline vec spherical2cartesian(const vec& x) {
    if (x.size() == 2)
      return vec_3(sin(x[0]) * cos(x[1]),
                   sin(x[0]) * sin(x[1]),
                   cos(x[0]));
    else if (x.size() == 3)
      return vec_3(x[2] * sin(x[0]) * cos(x[1]),
                   x[2] * sin(x[0]) * sin(x[1]),
                   x[2] * cos(x[0]));
    else
      assert(false);
  }

  /**
   * Convert spherical coordinates to Cartesian coordinates.
   * All angles are in degrees.
   * @param x  [theta, phi, radius], where theta in [0,180], phi in [-180,180]
   */
  inline vec spherical2cartesian_deg(const vec& x) {
    if (x.size() == 2)
      return spherical2cartesian(vec_2(degree2radian(x[0]),
                                       degree2radian(x[1])));
    else if (x.size() == 3)
      return spherical2cartesian(vec_3(degree2radian(x[0]),
                                       degree2radian(x[1]), x[2]));
    else
      assert(false);
  }

  /**
   * Given two angles in spherical coordinates, return the angle between them.
   * All angles are in radians.
   * @param theta1  For angle 1; in range [0,pi]
   * @param phi1    For angle 1; in range [-pi,pi]
   */
  inline double spherical_angle_diff(double theta1, double phi1,
                                     double theta2, double phi2) {
    using namespace std;
    return acos(sin(theta1) * cos(phi1) * sin(theta2) * cos(phi2) +
                sin(theta1) * sin(phi1) * sin(theta2) * sin(phi2) +
                cos(theta1) * cos(theta2));
  }

  /**
   * Given two angles in spherical coordinates, return the angle between them.
   * In this version, all angles are in degrees.
   * @param theta1  For angle 1; in range [0,180]
   * @param phi1    For angle 1; in range [-180,180]
   */
  inline double spherical_angle_diff_deg(double theta1, double phi1,
                                         double theta2, double phi2) {
    return radian2degree(spherical_angle_diff
                         (degree2radian(theta1), degree2radian(phi1),
                          degree2radian(theta2), degree2radian(phi2)));
  }

} // namespace sill

#endif // #ifndef _SILL_GEOMETRY_HPP_

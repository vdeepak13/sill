
#include <iostream>

#include <sill/math/geometry.hpp>

using namespace sill;
using namespace std;

void spherical_angle_test(double theta1, double phi1,
                          double theta2, double phi2);

int main(int argc, char** argv) {

  double theta1, phi1, theta2, phi2;

  theta1 = 0; phi1 = 0; theta2 = 0; phi2 = 90;
  spherical_angle_test(theta1, phi1, theta2, phi2);

  theta1 = 0; phi1 = 0; theta2 = 90; phi2 = 0;
  spherical_angle_test(theta1, phi1, theta2, phi2);

  theta1 = 90; phi1 = 0; theta2 = 90; phi2 = 0;
  spherical_angle_test(theta1, phi1, theta2, phi2);

  theta1 = 180; phi1 = -180; theta2 = 0; phi2 = 180;
  spherical_angle_test(theta1, phi1, theta2, phi2);

  theta1 = 90; phi1 = 90; theta2 = -90; phi2 = -90;
  spherical_angle_test(theta1, phi1, theta2, phi2);

  theta1 = 45; phi1 = 45; theta2 = -45; phi2 = -45;
  spherical_angle_test(theta1, phi1, theta2, phi2);
}

void spherical_angle_test(double theta1, double phi1,
                          double theta2, double phi2) {
  cout << "Diff btwn angles [" << theta1 << "," << phi1 << "] and ["
       << theta2 << "," << phi2 << "]: "
       << spherical_angle_diff_deg(theta1,phi1,theta2,phi2) << "\n";
}

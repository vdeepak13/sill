#include <iostream>
#include <prl/math/ublas/quaternion.hpp>
#include <prl/math/ublas/fixed.hpp>

typedef boost::numeric::ublas::fixed_vector<double,3> vector3;
typedef prl::quaternion<double> quaternion;

int main()
{
  using namespace std;
  vector3 v(0.7, 0.2, 0.5);
  quaternion q(1,2,3,4);
  q.normalize();

  quaternion q2(-0.933333333333333,0.133333333333333,0.2,0.266666666666667);
  vector3 qv(-0.0733333333333334,0.733333333333333,0.486666666666666);

  cout << (q*q) << endl;
  cout << q(v) << endl;

  cout << q.drotate(v) << endl;
  cout << q.dnormalize() << endl;

  assert(norm_2(q(v)-qv)<1e-10);
  assert(norm_2(q*q-q2)<1e-10);

  return 0;
}

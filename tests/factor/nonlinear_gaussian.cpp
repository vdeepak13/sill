#include <sill/base/universe.hpp>
#include <sill/factor/approx/hybrid_conditional.hpp>
#include <sill/factor/approx/integration_points.hpp>
#include <sill/factor/nonlinear_gaussian.hpp>
#include <sill/factor/operations.hpp>
#include <sill/math/function/linear.hpp>

int main() {
  using namespace sill;
  using namespace std;

  /*
  mat p;
  vec w;
  boost::tie(p, w) = approx.points(2);
  cout << p << endl;
  cout << w << endl;
  cout << sum(w) << endl;

  boost::tie(p, w) = approx.points(3);
  cout << p << endl;
  cout << w << endl;
  cout << sum(w) << endl;
  */

  universe u;
  vector_variable* x = u.new_vector_variable("x", 1);
  vector_variable* y = u.new_vector_variable("y", 1);
  vector_variable* z = u.new_vector_variable("z", 1);
  vector_var_vector xy = make_vector(x, y);
  vector_var_vector zv = make_vector(z);

  integration_points_approximator ip_approx;
  hybrid_conditional_approximator hybrid_approx(ip_approx, x, 25, 0, 3);

  // Create a prior and the CPDs (all of which are equivalent)
  moment_gaussian prior(xy, "1 2", "2 1; 1 2");
  moment_gaussian cpd_mg(zv, "1", "0", xy, "1 2");

  linear_vec fn("1 2", "1");
  nonlinear_gaussian cpd_ip(zv, xy, fn, ip_approx);
  nonlinear_gaussian cpd_hybrid(zv, xy, fn, hybrid_approx);

  // Create a prior and two equivalent CPDs: a linear and a nonlinear one
  moment_gaussian joint_mg = prior * cpd_mg;
  moment_gaussian joint_ip = prior * cpd_ip;
  moment_gaussian joint_hybrid = prior * cpd_hybrid;
  cout << joint_mg << endl;
  cout << joint_ip << endl;
  cout << joint_hybrid << endl;
}


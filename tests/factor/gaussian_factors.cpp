#include <iostream>

#include <boost/random/lagged_fibonacci.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/moment_gaussian.hpp>
#include <sill/factor/operations.hpp>

#define VERBOSE_ASSERT(X)                       \
  if (X) {                                      \
    std::cerr << "Error: " << #X << std::endl;  \
    return 1;                                   \
  }

int main()
{
  using namespace sill;
  using namespace std;

  universe u;
  vector_variable* x = u.new_vector_variable("x", 1);
  vector_variable* y = u.new_vector_variable("y", 1);
  vector_variable* z = u.new_vector_variable("z", 1);
  vector_variable* q = u.new_vector_variable("q", 1);

  vector_var_vector dom_xy  = make_vector(x, y);
  vector_var_vector dom_yx = make_vector(y, x);
  vector_var_vector dom_yz  = make_vector(y, z);

  mat ma  = mat_2x2(1.0, 2.0, 2.0, 3.0);
  mat ma_ = mat_2x2(3.0, 2.0, 2.0, 1.0);
  mat mb  = mat_2x2(2.0, 3.0, 3.0, 4.0);
  vec va  = vec_2(1.0, 2.0);
  vec va_ = vec_2(2.0, 1.0);
  vec vb  = vec_2(3.0, 4.0);

  canonical_gaussian f_xy_a_a(dom_xy, ma, va);
  canonical_gaussian f_yx_a_a(dom_yx, ma_, va_);
  canonical_gaussian f_yz_b_b(dom_yz, mb, vb);
  canonical_gaussian f_xy_b_b(dom_xy, mb, vb);
  canonical_gaussian f_xy_a_b(dom_xy, ma, vb);

  VERBOSE_ASSERT(f_xy_a_a != f_xy_a_a);
  VERBOSE_ASSERT(f_xy_a_a != f_yx_a_a);
  VERBOSE_ASSERT(f_xy_a_a == f_yz_b_b);
  VERBOSE_ASSERT(f_xy_a_a == f_xy_b_b);
  VERBOSE_ASSERT((f_xy_a_a < f_yz_b_b) != (x<y));
  VERBOSE_ASSERT(!(f_xy_a_a < f_xy_b_b));
  VERBOSE_ASSERT(f_xy_b_b < f_xy_a_b);
  /*
  if (f_xy_a_a != f_xy_a_a) {
    std::cerr << "f_xy_a_a != f_xy_a_a" << std::endl;
    return 1;
  }
  if (f_xy_a_a != f_yx_a_a) {
    std::cerr << "f_xy_a_a != f_yx_a_a" << std::endl;
    return 1;
  }
  if (f_xy_a_a == f_yz_b_b) {
    std::cerr << "f_xy_a_a == f_yz_b_b" << std::endl;
    return 1;
  }
  if (f_xy_a_a == f_xy_b_b) {
    std::cerr << "f_xy_a_a == f_xy_b_b" << std::endl;
    return 1;
  }
  cout << ((f_xy_a_a < f_yz_b_b) == (x<y)) << endl;
  if (f_xy_a_a >= f_xy_b_b) {
    std::cerr << "" << std::endl;
    return 1;
  }
  if (f_xy_b_b < f_xy_a_b) {
    std::cerr << "" << std::endl;
    return 1;
  }
  */

  vector_assignment assign;
  assign[y] = zeros(1);
  assign[z] = zeros(1);

  cout << "f_xy_a_a = " << f_xy_a_a << endl;
  cout << "f_yz_b_b = " << f_yz_b_b << endl;
  cout << "f_xy_a_a * f_yz_b_b = " << (f_xy_a_a*f_yz_b_b) << endl;
  cout << "restrict(f_xy_a_a * f_yz_b_b, [y=1,z=1]) = "
       << (f_xy_a_a * f_yz_b_b).restrict(assign) << endl;
  cout << "f_xy_a_a.marginal{x}) = "
       << (f_xy_a_a * f_yz_b_b).marginal(make_domain(x)) << endl;

  vector_var_map vm;
  vm[x] = z;
  vm[y] = q;
  f_xy_a_a.subst_args(vm);
  cout << f_xy_a_a << endl;

  ma(1,1) = 5;
  moment_gaussian mg(dom_xy, va, ma); // [1 2], [1 2; 2 5]
  vec val = vec_2(0.5, 0.5);
  cout << "mg = " << mg << endl;
  cout << "mg(" << val << ") = " << mg(val) << endl;
  
  vector_assignment asg; 
  asg[y] = ones(1);
  cout << "mg.restrict(y=1)" << mg.restrict(asg) 
       << endl;
  // -1.8236574894
  //cout << "mg == mg: " << (mg == mg) << endl;

  // Test conditional moment gaussians.
  mg = moment_gaussian(dom_xy, va, ma, make_vector(z,q), mb);
  cout << "mg = " << mg << endl;
  asg.clear();
  asg[x] = vec(".5");
  asg[y] = vec(".5");
  asg[z] = ones(1);
  asg[q] = ones(1);
  cout << "mg.restrict(x=.5,y=.5,z=1,q=1)" << mg.restrict(asg) << endl;
  asg.clear();
  asg[z] = ones(1);
  asg[q] = ones(1);
  mg = mg.restrict(asg);
  cout << "mg.restrict(z=1,q=1)" << mg << endl;
  cout << "mg.restrict(z=1,q=1)(" << val << ") = " << mg(val) << endl;

  canonical_gaussian cg =   //(make_domain(dom_xy), 1);
    moment_gaussian(make_vector(x), zeros(1), eye(1,1), 
                    make_vector(y), ones(1,1));
  cout << "cg = " << cg << endl;

  // Test sampling, conditioning, and restricting.
  moment_gaussian mg_xy(dom_xy, va, ma); // [1 2], [1 2; 2 5]
  moment_gaussian mg_x_given_y(mg_xy.conditional(make_domain(y)));
  moment_gaussian mg_y(mg_xy.marginal(make_domain(y)));
  //  boost::lagged_fibonacci607 rng(3467134);
  boost::mt11213b rng(2359817);
  double mg_xy_ll(0);
  double mg_x_given_y_ll(0);
  double mg_y_ll(0);
  size_t nsamples = 100;
  for (size_t i(0); i < nsamples; ++i) {
    vector_assignment va(mg_xy.sample(rng));
    mg_xy_ll += mg_xy.logv(va);
    vector_assignment va_y;
    va_y[y] = va[y];
    mg_x_given_y_ll += mg_x_given_y.restrict(va_y).logv(va);
    mg_y_ll += mg_y.logv(va);
  }
  mg_xy_ll /= nsamples;
  mg_x_given_y_ll /= nsamples;
  mg_y_ll /= nsamples;

  cout << "\nSampled " << nsamples << " samples from P(x,y)\n"
       << "  Computed E[log P(x,y)] = " << mg_xy_ll << "\n"
       << "  Computed E[log P(x|y)] = " << mg_x_given_y_ll << "\n"
       << "  Computed E[log P(y)] = " << mg_y_ll << "\n"
       << "  Computed E[log P(x|y)] + E[log P(y)] = "
       << (mg_x_given_y_ll + mg_y_ll) << "\n"
       << endl;
}

#define BOOST_DISABLE_ASSERTS
#define NDEBUG

#include <iostream>
#include <vector>

#include <prl/factor/fixed_factors.hpp>
#include <prl/factor/random.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/timer.hpp>

#include <prl/macros_def.hpp>

boost::mt19937 rng;

int main(int argc, char** argv) {
  using namespace prl;
  using namespace std;

  size_t n = (argc<2) ? 1000 : boost::lexical_cast<size_t>(argv[1]);

  universe univ;
  std::vector<finite_variable*> vars = univ.new_finite_variables(2, 2);
  boost::uniform_int<> unif(0,1); // uniform from {0,1}

  /*
  // Test correctness
  for(size_t i = 0; i < 100; i++) {
    variable* v = vars[unif(rng)];
    domain d(vars);
    unary_factor<2>  f  = random_discrete_factor< unary_factor<2> >(v, rng);
    binary_factor<2> g  = random_discrete_factor< binary_factor<2> >(d, rng);
    binary_factor<2> fg = g*f;
    unary_factor<2>  h  = fg.marginal(v);

    tablef ft = f, gt = g, fgt = ft*gt, ht = fgt.marginal(v);


//     cout << "f=" << f << endl;
//     cout << "g=" << g << endl;
//     cout << "f*g=" << fg << endl;
//     cout << "sum(f*g)=" << h << endl;

//     cout << "ft=" << ft << endl;
//     cout << "gt=" << gt << endl;
//     cout << "ft*gt=" << fgt << endl;
//     cout << "sum(ft*gt)=" << ht << endl;

    double fg_diff = norm_inf(tablef(fg), fgt);
    double h_diff = norm_inf(tablef(h), ht);
    assert(fg_diff < 1e-10 && h_diff < 1e-10);
  }
*/

  // Test timings
  std::vector< unary_factor<2> > unary(1000);
  std::vector< binary_factor<2> > binary(1000);
  for(size_t i = 0; i < 1000; i++) {
    finite_variable* v = vars[unif(rng)];
    finite_domain d(vars);
    unary[i] = random_discrete_factor< unary_factor<2> >(v, rng);
    binary[i] = random_discrete_factor< binary_factor<2> >(d, rng);
  }

  boost::timer t;
  double dummy = 0;

  unary_factor<2> f;

  t.restart();
  for(size_t j = 0; j < n; j++)
    for(size_t i = 0; i < 1000; i++) {
      f *= unary[i];
      //dummy += unary[i].norm_constant();
      // dummy += combine(unary[i], unary[i], product_op).norm_constant();
    }
  dummy = f[0]+f[1];
  cout << "unary * unary: " << (n/t.elapsed()/1e3) << " MIPS" << endl;

//   t.restart();
//   for(size_t j = 0; j < n; j++)
//     for(size_t i = 0; i < 1000; i++)
//       dummy += (unary[i]*binary[i])(0,0);//.norm_constant();
//   cout << "binary * unary: " << (n/t.elapsed()/1e3) << " MIPS" << endl;

//   t.restart();
//   for(size_t j = 0; j < n; j++)
//     for(size_t i = 0; i < 1000; i++)
//       dummy += binary[i].marginal(vars[0])[0];//.norm_constant();
//   cout << "binary.collapse(): " << (n/t.elapsed()/1e3) << " MIPS" << endl;

  cout << dummy << endl;

}

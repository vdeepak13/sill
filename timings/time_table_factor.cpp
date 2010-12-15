#include <iostream>
#include <string>
#include <iterator>
#include <cmath>
#include <map>

#include <boost/array.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/timer.hpp>

#include <prl/base/universe.hpp>
#include <prl/math/gdl_enum.hpp>
#include <prl/factor/table_factor.hpp>
#include <prl/copy_ptr.hpp>
#include <prl/factor/random.hpp>

#include <prl/range/algorithm.hpp>

using namespace prl;

int main(int argc, char** argv) {

  using namespace boost;
  using namespace prl;
  using namespace std;
  
  // The number of repetitions
  assert(argc==2);
  size_t N = boost::lexical_cast<size_t>(argv[1]);

  // Create a source of random numbers.
  boost::mt19937 rng;
  uniform_real<> unif;

  // Create the variables
  universe u;
  finite_variable* w = u.new_finite_variable(4);
  finite_variable* x = u.new_finite_variable(3);
  finite_variable* y = u.new_finite_variable(5);
  const size_t maxn = 40;

  // Generate some random numbers
  boost::array<double, 3*4*5*maxn> val;
  for(size_t i = 0; i < val.size(); i++) val[i] = unif(rng);

  cout << "Join operations on factors with 2 arguments: " << endl;
  double sum = 0;
  for(size_t n = 1; n < maxn; n += 5) {
    finite_variable* z = u.new_finite_variable(n);
    finite_domain xz = make_domain(x, z);
    finite_domain yz = make_domain(y, z);
    
    timer t;
    for(size_t i = 0; i < N; i++) {
//      cout << i << endl;
      table_factor f(xz, 0);
      table_factor g(yz, 0);
      std::copy(val.begin(), val.begin()+f.size(), f.values().first);
      std::copy(val.begin(), val.begin()+g.size(), g.values().first);
      table_factor h = f*g;
      sum += h.norm_constant();
    }
    
    cout << N << " joins [3," << n << "] x [" << n << ", 5]: " 
         << t.elapsed() << "s" << endl;
  }

  cout << "Join operations on factors with 3 arguments: " << endl;
  for(size_t n = 1; n < maxn; n += 5) {
    finite_variable* z = u.new_finite_variable(n);
    finite_domain wxz = make_domain(w, x, z);
    finite_domain wyz = make_domain(w, y, z);
    
    timer t;
    for(size_t i = 0; i < N; i++) {
      table_factor f(wxz, 0);
      table_factor g(wyz, 0);
      std::copy(val.begin(), val.begin()+f.size(), f.values().first);
      std::copy(val.begin(), val.begin()+g.size(), g.values().first);
      table_factor h = f*g;
      sum += h.norm_constant();
    }
    
    cout << N << " joins [3,4," << n << "] x [" << n << ",4,5]: " 
         << t.elapsed() << "s" << endl;
  }

  cout << sum << endl;
}

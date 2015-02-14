#include <sill/math/function/softmax.hpp>

#include <boost/timer.hpp>

int main(int argc, char** argv) {
  using namespace sill;
  using namespace std;
  typedef softmax<double>::mat_type mat_type;
  typedef softmax<double>::vec_type vec_type;
  
  size_t nl = atoi(argv[1]);
  size_t nf = atoi(argv[2]);
  size_t ni = atoi(argv[3]);

  boost::timer t;
  softmax<double> f(nl, nf);
  vec_type x(nf);
  t.restart();
  for (size_t i = 0; i < ni; ++i) {
    vec_type p = f(x);
  }
  cout << "Value: " << t.elapsed() << endl;

  t.restart();
  for (size_t i = 0; i < ni; ++i) {
    vec_type p = f.log(x);
  }
  cout << "Log-value: " << t.elapsed() << endl;

  return 0;
}


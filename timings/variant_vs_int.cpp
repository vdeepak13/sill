#include <boost/variant.hpp>
#include <boost/timer.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <vector>
#include <iostream>

int main(int argc, char** argv)
{
  using namespace std;
  int l = argc>1 ? atol(argv[1]) : 10000;
  const int k = 10000;
  std::vector<int> intv(k);
  std::vector< boost::variant<int> > varv(k);

  boost::mt19937 rng;
  for (int j = 0; j < k; j++) varv[j] = intv[j] = rng();

  boost::timer t;
  int c = 0;
  for (int i = 0; i < l; i++) {
    for (int j = 0; j < k; j++) c += intv[j];
  }
  cout << "vector<int>: " << t.elapsed() << ' ' << c << endl;

  c = 0;
  t.restart();
  for (int i = 0; i < l; i++) {
    for (int j = 0; j < k; j++) c += boost::get<int>(varv[j]);
  }
  cout << "vector<variant>: " << t.elapsed() << ' ' << c << endl;
  cout << "sizeof(variant<int>) = " << sizeof(varv[0]) << endl;
}

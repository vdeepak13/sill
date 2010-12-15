#define NDEBUG
#define BOOST_DISABLE_ASSERTS

#include <boost/array.hpp>
#include <boost/timer.hpp>
#include <vector>
#include <iostream>

#include <boost/random/mersenne_twister.hpp>

boost::mt19937 rng;

int main()
{
  using namespace std;
  size_t n= 50000;
  boost::array<double, 1000> x;

  for(size_t i = 0; i < 1000; i++) {
    x[i] = rng();
  }

  boost::timer t;
  std::plus<double> op;
  double dummy;
  t.restart();
  for(size_t j = 0; j < n; j++)
    for(size_t i = 0; i < 1000; i++) {
      dummy += op(x[i],x[i]);
    }

  cout << "unary * unary: " << (n/t.elapsed()/1e3) << " MIPS" << endl;
  cout << dummy << endl;

}


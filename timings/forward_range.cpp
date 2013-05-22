#include <iostream>
#include <vector>
#include <iterator>

#include <boost/random/mersenne_twister.hpp>
#include <boost/timer.hpp>

#include <sill/range/forward_range.hpp>

#include <sill/macros_def.hpp>

using sill::forward_range;

forward_range<int> getrange(const std::vector<int>& v) {
  return v;
}

int f(int x) {
  return x;
  int result = 1;
  for(int i=0; i<10; i++) result*=x++;
  return result;
}

int g(const forward_range<const int&>& values) {
  return *boost::begin(values);
}

int main(int argc, char** argv) {
  using namespace std;
  using namespace boost;

  boost::mt19937 rng;
  assert(argc==2);

  int m = atoi(argv[1]);
  int n = 1000;
  
  std::vector<int> v(n);
  for(int i=0; i<n; i++) v[i] = rng();

  int result=0;
  
  timer t;
  for(int i = 0; i < m; i++)
    for(int j = 0; j < n; j++) result += f(v[j]);
  cout << "Direct access: " << (long(m)*n/t.elapsed()/1e6) << "MIPS" << endl;

  t.restart();
  for(int i = 0; i < m; i++) {
    foreach(int x, v) result += f(x);
  }
  cout << "Foreach: " << (long(m)*n/t.elapsed()/1e6) << "MIPS" << endl;

  t.restart();
  for(int i = 0; i < m; i++) {
    foreach(int x, getrange(v)) result += f(x);
  }
  cout << "Any range + foreach: " 
       << (long(m)*n/t.elapsed()/1e6) << "MIPS" << endl;

  t.restart();
  forward_range<const int&> ar(v);
  for(int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) 
      result += g(ar);
  cout << "Any range 1 invocation " 
       << (long(m)*n/t.elapsed()/1e6) << "MIPS" << endl;

  t.restart();
  for(int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) 
      result += g(forward_range<const int&>(v));
  cout << "Any range construction + 1 invocation " 
       << (long(m)*n/t.elapsed()/1e6) << "MIPS" << endl;

  cout << result << endl;
}

/*
Direct access: 1250MIPS
Foreach: 434.783MIPS
Any range + foreach: 13.3333MIPS
*/


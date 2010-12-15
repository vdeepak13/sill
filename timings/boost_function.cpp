// tests boost::function

#include <boost/timer.hpp>
#include <boost/function.hpp>
#include <boost/lexical_cast.hpp>
#include <functional>
#include <iostream>
boost::function<int (int,int)> f;

int main(int argc, char** argv) {
  using namespace boost;
  using namespace std;

  assert(argc == 3);
  int n = lexical_cast<int>(argv[1]);
  string which(argv[2]);
  if (which == "plus") f = std::plus<int>();
  else f = std::minus<int>();
  
  int sum = 0;

  boost::timer t;
  
  for(int i=0; i<1000; i++)
    for(int j=0; j<n; j++) 
      sum += f(i, j);
  
  cout << "boost::function calls " << (n/t.elapsed()/1e3) << "MIPS" << endl;
  cout << sum << endl;
}

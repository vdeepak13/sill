#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>

#include <boost/timer.hpp>

enum op_type { PLUS = 1, MINUS = 2, TIMES = 3, DIV = 4};

template <typename T>
T f(const std::vector<T>& v, op_type op) {
  T result = 1;
  for(std::size_t i = 0; i<v.size(); i++)
    switch (op) {
    case PLUS: result += v[i]; result += v[i]; break;
    case MINUS: result -= v[i]; result += v[i]; break;
    case TIMES: result *= v[i]; result += v[i]; break;
    case DIV: result /= v[i]; result += v[i]; break;
    default: assert(false);
    }
  return result;
}

template <typename T>
T g(const std::vector<T>& v, op_type op) {
  T result = 0;
  switch (op) {
  case PLUS: for(std::size_t i = 0; i < v.size(); i++)  {
      result += v[i]; result += v[i];
    } break;
  case MINUS: for(std::size_t i = 0; i < v.size(); i++) {
      result -= v[i]; result += v[i];
    } break;
  case TIMES: for(std::size_t i = 0; i < v.size(); i++) {
      result *= v[i]; result += v[i];
    } break;
  case DIV: for(std::size_t i = 0; i < v.size(); i++) {
      result /= v[i]; result += v[i];
    } break;
  default: assert(false);//result /= 0;
  }
  return result;
}

int main(int argc, char** argv)
{
  using namespace std;
  typedef double storage_type;

  assert(argc>2);
  size_t n = atoi(argv[1]);
  op_type op = op_type(atoi(argv[2]));
  std::vector<storage_type> v(1000);
  for(size_t i = 0; i < v.size() ; i++) v[i] = rand();
  cout << "Op: " << op << endl;

  storage_type result = 0;
  boost::timer t;
  for(size_t i = 0; i < n; i++) 
    result += f(v, op);
  cout << "Enum: " << t.elapsed() / 1000 / n << endl;
    
  t.restart();
  for(size_t i = 0; i < n ; i++)
    result += g(v, op);
  cout << "Direct: " << t.elapsed() / 1000 / n << endl;

  cout << result << endl;
  return 0;
}

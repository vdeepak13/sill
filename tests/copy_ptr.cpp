#include <iostream>
#include <sill/copy_ptr.hpp>
#include <boost/timer.hpp>
#include <boost/lexical_cast.hpp>

struct test {
  double a, b;
  double f() const { return a + b; }
};

// Speed test for shared pointer operations
int main(int argc, char** argv)
{
  using namespace sill;
  using namespace std;
  using namespace boost;

  assert(argc==2);
  size_t n = lexical_cast<size_t>(argv[1]);

  double result = 0.;
  test* raw = new test();

  timer t;
  for(size_t i = 0; i < n; i++) 
    for(size_t j = 0; j < 1000; j++) {
      //test* raw2 = raw;
      result += raw->f();
    }
  cout << "Raw pointer: " << n/t.elapsed()/1e3 << " MIPS." << endl;
  cout << result << endl;
  
  shared_ptr<test> shared(new test());
  t.restart();
  for(size_t i = 0; i < n; i++) 
    for(size_t j = 0; j < 1000; j++) {
      shared_ptr<test> shared2 = shared;
      result += shared2->f();
    }
  cout << "Shared pointer: " << n/t.elapsed()/1e3 << " MIPS." << endl;
  cout << result << endl;

  copy_ptr<test> copy(new test());
  t.restart();
  for(size_t i = 0; i < n; i++)
    for(size_t j = 0; j < 1000; j++) {
      //const copy_ptr<test> copy2 = copy;
      result += copy->f();
    }
  cout << "Copy pointer: " << n/t.elapsed()/1e3 << " MIPS." << endl;
  cout << result << endl;

  const copy_ptr<test> const_copy(new test());
  t.restart();
  for(size_t i = 0; i < n; i++)
    for(size_t j = 0; j < 1000; j++)
      result += const_copy->f();
  cout << "Copy pointer (const access): " << n/t.elapsed()/1e3 << " MIPS." << endl;
  cout << result << endl;

}


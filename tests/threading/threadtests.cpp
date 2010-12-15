
#include <stdlib.h>
#include <iostream>
#include <vector>

#include <prl/parallel/pthread_tools.hpp>

using namespace prl;
using namespace std;


class fib : public thread {
  size_t fib_fun(size_t x) {
    if(x < 2) return x; 
    else return fib_fun(x-1) + fib_fun(x-2);
  }
  size_t m_input;
  size_t m_output;
public:
  fib(size_t input = 0) : m_input(input) { }
  void set(size_t input) { m_input = input; }
  size_t output() { return m_output; }
  void run() {
    cout << "Running fib(" << m_input << ")" << endl;
    m_output = fib_fun(m_input);
  }  
};



int main(int argc, char* argv[]) {
  cout << "Testing Thread and ThreadGroup." << endl;
  size_t count = 50;
  vector<fib> fibs(count);
  // Launch a bunch of threads
  for(size_t i = 0; i < fibs.size(); ++i) {
    cout << "Luanching fib " << i << endl;
    fibs[i].set(i);
    fibs[i].start();
  }

  // Join all the threads
  for(size_t i = 0; i < fibs.size(); ++i) {
    fibs[i].join();
    cout << "fib(" << i << ") = " << fibs[i].output() << endl;
  }
  
  return EXIT_SUCCESS;
}



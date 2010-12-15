#include <iostream>

#include <boost/lexical_cast.hpp>
#include <boost/timer.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

#include <itpp/itbase.h>

int main(int argc, char** argv) {

  namespace ublas = boost::numeric::ublas;
  using namespace itpp;
  using namespace std;

  size_t m = boost::lexical_cast<size_t>(argv[1]);  
  size_t n = boost::lexical_cast<size_t>(argv[2]);

  typedef ublas::matrix<double> umat;
  typedef ublas::vector<double> uvec;

  boost::timer t;
  double x;
  
  for(size_t size = 10; size <= 50; size += 10) {
    mat a(size, size);
    mat b(size, size);
    
    t.restart();
    for(size_t i = 0; i < m; i++) {
      mat c = a * b;
      x += c(0,0);
    }
    cout << size << "x" << size << " matrix multiplication with IT++ took "
         << t.elapsed() / m << "s" << endl;
    
    umat ua(size, size);
    umat ub(size, size);
    t.restart();
    for(size_t i = 0; i < m; i++) {
      umat c = prod(ua, ub);
      x += c(0,0);
    }
    cout << size << "x" << size << " matrix multiplication with uBLAS took "
         << t.elapsed() / m << "s" << endl;
  }

  for(size_t size = 10; size <= 50; size += 10) {
    mat a(size, size);
    vec b(size);
    
    t.restart();
    for(size_t i = 0; i < n; i++) {
      vec c = a * b;
      x += c(0);
    }
    cout << size << "x" << size << " matrix-vector multiplication w/ IT++ took "
         << t.elapsed() / n << "s" << endl;
    
    umat ua(size, size);
    uvec ub(size);
    t.restart();
    for(size_t i = 0; i < n; i++) {
      uvec c = prod(ua, ub);
      x += c(0);
    }
    cout << size << "x" << size << " matrix-vector multiplication w/uBLAS took "
         << t.elapsed() / n << "s" << endl;
  }
  
  cout << x << endl;
}

/*
10x10 matrix multiplication with IT++ took 0s
10x10 matrix multiplication with uBLAS took 0.00021s
20x20 matrix multiplication with IT++ took 5e-06s
20x20 matrix multiplication with uBLAS took 0.001505s
30x30 matrix multiplication with IT++ took 1e-05s
30x30 matrix multiplication with uBLAS took 0.005005s
40x40 matrix multiplication with IT++ took 1.5e-05s
40x40 matrix multiplication with uBLAS took 0.01131s
50x50 matrix multiplication with IT++ took 3e-05s
50x50 matrix multiplication with uBLAS took 0.022635s
10x10 matrix-vector multiplication w/ IT++ took 0s
10x10 matrix-vector multiplication w/uBLAS took 8e-06s
20x20 matrix-vector multiplication w/ IT++ took 0s
20x20 matrix-vector multiplication w/uBLAS took 2.9e-05s
30x30 matrix-vector multiplication w/ IT++ took 0s
30x30 matrix-vector multiplication w/uBLAS took 5.2e-05s
40x40 matrix-vector multiplication w/ IT++ took 2e-06s
40x40 matrix-vector multiplication w/uBLAS took 9e-05s
50x50 matrix-vector multiplication w/ IT++ took 1e-06s
50x50 matrix-vector multiplication w/uBLAS took 0.000142s
*/

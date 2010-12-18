#include <iostream>
#include <string>
#include <algorithm>

#include <boost/array.hpp>

#include <sill/base/finite_assignment.hpp>
#include <sill/base/finite_assignment_iterator.hpp>
#include <sill/base/vector_assignment.hpp>
#include <sill/base/universe.hpp>

int main(int argc, char** argv) {

  using namespace sill;
  using namespace std;

  // Create a universe.
  universe u;

  // Create some variables.
  finite_variable* v = u.new_finite_variable(2);
  finite_variable* x = u.new_finite_variable(3);
  vector_variable* y = u.new_vector_variable(2);
  
  // Create an assignment.
  finite_assignment a_f;
  vector_assignment a_v;
  a_f[x] = 2;
  a_v[y] = vector_variable::value_type(2);

  cout << x << ": " << a_f[x] << endl;
  cout << y << ": " << a_v[y] << endl;

  vector_variable::value_type val = a_v[y];
  cout << x << ": " << a_f[x] << endl;
  cout << y << ": " << val << endl;

  // Test the assignment iterator
  boost::array<finite_variable*, 2> d = {{ v, x }};
  finite_assignment_iterator it(d), end;

  cout << "Assignments: " << endl;
  while (it!=end)
    cout << *it++ << endl;

  // Test the empty iterator
  boost::array<finite_variable*, 0> empty;
  it = finite_assignment_iterator(empty);
 
  cout << "Empty assignment: " << *it << endl;
  cout << "it == end: " << (it==end) << endl;
  cout << "++it == end: " << (++it==end) << endl;

  return EXIT_SUCCESS;
}

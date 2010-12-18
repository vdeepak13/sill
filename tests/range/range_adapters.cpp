#include <iostream>
#include <vector>

#include <boost/array.hpp>

#include <sill/functional.hpp>
#include <sill/range/algorithm.hpp>
#include <sill/range/transformed.hpp>
#include <sill/range/reversed.hpp>
#include <sill/range/joined.hpp>
#include <sill/range/forward_range.hpp>
#include <sill/range/io.hpp>

int main()
{
  using namespace std;
  using namespace sill;
  boost::array<int,4> a = {{0, 2, 4, 8}};
  boost::array<int,2> b = {{0, 2}};
  std::vector<int> v;
  
  copy(a, back_inserter(v));
  
  cout << a << endl;
  cout << make_transformed(a, identity_t<int>()) << endl;
  cout << make_reversed(a) << endl;
  cout << make_joined(a, b) << endl;
  cout << forward_range<int>(a) << endl;

  forward_range<int> fr = forward_range<int>(make_joined(a, b));
  cout << fr << endl;
}

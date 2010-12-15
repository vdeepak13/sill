#include <iostream>
#include <vector>

#include <boost/array.hpp>

#include <prl/functional.hpp>
#include <prl/range/algorithm.hpp>
#include <prl/range/transformed.hpp>
#include <prl/range/reversed.hpp>
#include <prl/range/joined.hpp>
#include <prl/range/forward_range.hpp>
#include <prl/range/io.hpp>

int main()
{
  using namespace std;
  using namespace prl;
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

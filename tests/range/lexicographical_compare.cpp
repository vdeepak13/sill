#include <prl/range/algorithm.hpp>
#include <boost/array.hpp>
#include <iostream>

#include <prl/global.hpp>

int main()
{
  using namespace prl;
  using namespace std;
  boost::array<int, 3> a = {{1, 3, 4}};
  boost::array<int, 3> b = {{1, 2, 4}};

  cout << "a<b: " << prl::lexicographical_compare(a, b) << endl;
  cout << "b<a: " << prl::lexicographical_compare(b, a) << endl;
  cout << "a<a: " << prl::lexicographical_compare(a, a) << endl;
}

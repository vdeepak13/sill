#include <sill/range/algorithm.hpp>
#include <boost/array.hpp>
#include <iostream>

#include <sill/global.hpp>

int main()
{
  using namespace sill;
  using namespace std;
  boost::array<int, 3> a = {{1, 3, 4}};
  boost::array<int, 3> b = {{1, 2, 4}};

  cout << "a<b: " << sill::lexicographical_compare(a, b) << endl;
  cout << "b<a: " << sill::lexicographical_compare(b, a) << endl;
  cout << "a<a: " << sill::lexicographical_compare(a, a) << endl;
}

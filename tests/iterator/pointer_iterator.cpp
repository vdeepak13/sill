#include <iostream>
#include <sill/iterator/pointer_iterator.hpp>

struct test {
  int a;
  int b;
  int c;
} t;


int main()
{
  using namespace std;
  using namespace sill;
  pointer_iterator<int, std::forward_iterator_tag> begin(&t.a);
  pointer_iterator<int, std::forward_iterator_tag> end((&t.c)+1);
  cout << end - begin << endl;
  cout << (begin < end) << endl;
  cout << (begin > end) << endl;
}

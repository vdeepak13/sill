#include <iostream>
#include <set>
#include <sill/iterator/subset_iterator.hpp>
#include <sill/base/stl_util.hpp>
/**
 * Test of the subset_iterator.
 */
int main(int argc, char** argv) {

  using namespace sill;

  typedef std::set<char> char_set_t;

  char_set_t s;
  s.insert('a');
  s.insert('b');
  s.insert('c');
  s.insert('d');

  std::cout << "Original set: " << s << std::endl;

  std::cout << "Strict subsets, minus empty set:" << std::endl;

  subset_iterator<char_set_t> it(s,1,s.size()-1);
  subset_iterator<char_set_t> end;
  while (it != end) {
    std::cout << *it << std::endl;
    ++it;
  }

  std::cout << "All subsets:" << std::endl;
  it = subset_iterator<char_set_t>(s);
  while (it != end) {
    std::cout << *it << std::endl;
    ++it;
  }

  std::cout << "Subsets of size <= 2:" << std::endl;
  it = subset_iterator<char_set_t>(s, 0, 2);
  while (it != end) {
    std::cout << *it << std::endl;
    ++it;
  }

  // Return success.
  return EXIT_SUCCESS;
}

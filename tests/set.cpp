#include <iostream>
#include <assert.h>
#include <prl/set.hpp>

int main(int argc, char** argv) {
  using namespace prl;

  typedef prl::set<int> set_type;

  set_type s1, s2, s3;
  for (int i = 1; i < 5; i++)
    s1.insert(i);
  for (int i = 3; i < 7; i++)
    s2.insert(i);

  std::cout << "S1: " << s1 << std::endl;

  std::cout << "S2: " << s2 << std::endl;

  std::cout << "S1 && S2: " << s1.intersect(s2) << std::endl;

  std::cout << "|S1 && S2|: " << s1.intersection_size(s2) << std::endl;

  std::cout << "S1 || S2: " << (s3 = s1.plus(s2)) << std::endl;

  std::cout << "S2 - S1: " << s2.minus(s1) << std::endl;

  assert(s1.intersection_size(s2) == 2);
  assert(s1.subset_of(s3));
  assert(s2.subset_of(s3));
  assert(s2.minus(s1).disjoint_from(s1));

  set_type a1, a2;
  boost::tie(a1, a2) = s1.partition(s2);
  assert(a1 == s1.intersect(s2));
  assert(a2 == s2.minus(s1));
  boost::tie(a1, a2) = s2.partition(s1);
  assert(a1 == s2.intersect(s1));
  assert(a2 == s1.minus(s2));

  // Return success.
  return EXIT_SUCCESS;
}

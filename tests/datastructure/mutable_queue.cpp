#include <iostream>
#include <queue>
#include <assert.h>
#include <boost/random/mersenne_twister.hpp>
#include <prl/datastructure/mutable_queue.hpp>
#include <boost/lexical_cast.hpp>

int main(int argc, char* argv[]) {

  boost::mt19937 rng;

  int n = (argc == 2) ? boost::lexical_cast<int>(argv[1]) : 1000;

  // Insert the same keys into STL queues and PRL queues and make sure
  // they spit out the keys in the same order.
  std::priority_queue<int> std_pq;
  prl::mutable_queue<int, int> prl_pq;
  for (int i = 0; i < n; i++) {
    int x = rng();
    int y = rng();
    prl_pq.push(x, y);
    std_pq.push(y);
    assert(std_pq.size() == prl_pq.size());
  }
  while (!std_pq.empty()) {
    assert(!prl_pq.empty());
    std::pair<int,int> xy;
    int std_y = std_pq.top();
    xy = prl_pq.pop();
    assert(xy.second == std_y);
    std_pq.pop();
  }

  // Insert a bunch of items into the PRL queue, reprioritize them,
  // and then make sure they come out in sorted order.
  for (int i = 0; i < n; i++) {
    int x = rng();
    prl_pq.push(i, x);
  }
  for (int i = 0; i < n; i++) {
    int x = rng();
    prl_pq.update(i, x);
  }
  int x = prl_pq.pop().second;
  while (!prl_pq.empty()) {
    int y = prl_pq.pop().second;
    assert(y <= x);
    x = y;
  }

  return EXIT_SUCCESS;
}

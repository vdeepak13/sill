#include <iostream>
#include <tr1/unordered_map>

#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/timer.hpp>

int main(int argc, char** argv)
{
  using namespace boost;
  using namespace std;
  using namespace std::tr1;

  unordered_map<int, void*> map;
  typedef std::pair<const int,void*> entry;

  int m = (argc > 1) ? lexical_cast<int>(argv[1]) : 30000;
  int n = 1000;
  int sum = 0;

  for(int i = 0; i < n; i++) map[i] = 0;

  boost::timer t;
  for(int i = 0; i < m; i++) {
    BOOST_FOREACH(entry& e, map) sum += e.first;
  }

  cout << (m/t.elapsed()) << " traversals per second" << endl;
}

#include <iostream>
#include <stdlib.h>
#include <tr1/unordered_map>

using namespace std;
// this is deprecated use boost unordered map instead

int main(int argc, char** argv) {
#ifdef __DARWIN__
  cout << "Hello World" << endl;
#endif
  
  typedef std::tr1::unordered_map<size_t, size_t> map_type;
  map_type map;
  map[1] = 2;
  map[2] = 3;
  map[3] = 4;

  map_type::const_iterator iter = map.find(1);
  if(iter != map.end()) {
    cout << "The end" << endl;
  }


  return EXIT_SUCCESS;
}

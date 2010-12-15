#include <iostream>
#include <vector>
#include <boost/lexical_cast.hpp>

#include <prl/distributed/address_functions.hpp>
#include <prl/distributed/chord_predecessor.hpp>
#include <prl/distributed/sfslite_io.hpp> // for str & bigint ostream output

#include <chord_types.h>
#include <id_utils.h>

int main (int argc, char *argv[]) {
  using namespace std;

  if (argc < 3) {
    cerr << "Usage: chord_predecessor host port" << endl;
    return -1;
  }
 
  std::string ip_address = prl::address_string(argv[1]);
  unsigned short port    = boost::lexical_cast<unsigned short>(argv[2]);

  chord_node prev;
  prev.r.hostname = ip_address.c_str();
  prev.r.port = port;
  prev.x = make_chordID(prev.r.hostname, prev.r.port);
  prev.vnode_num = 0;

  chordID first = prev.x;
  cout << "First node: " << first << endl;
  
  do {
    prl::chord_predecessor chord(prev);
    prev = chord.predecessor_node();
    cout << "Previous node: " << prev.x << endl;
  } while(prev.x != first);

}

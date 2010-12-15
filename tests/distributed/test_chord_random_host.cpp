#include <iostream>

#include <boost/lexical_cast.hpp>
#include <boost/thread/thread.hpp>

#include <prl/distributed/chord_random_host.hpp>
#include <prl/distributed/address_functions.hpp>

int main(int argc, char** argv) {
  using namespace std;
  using namespace prl;

  if (argc < 3) { 
    cerr << "Usage: test_chord_random_host hostname port\n\n"
	 << "hostname and port specifies the location of the Chord service.\n"
	 << "hostname must not be localhost / 127.0.0.1\n";
    return EXIT_FAILURE;
  }

  std::string ip_address = address_string(argv[1]);
  unsigned short port = boost::lexical_cast<int32_t>(argv[2]);
  chord_random_host hostfn(ip_address, port, 0);

  cout << "Random hosts:" << endl;
  for(size_t i = 0; i < 10; i++) 
    cout << i << ": " << hostfn().first << endl;

  return EXIT_SUCCESS;
}

#include <iostream>
#include <cstdlib>

#include <boost/lexical_cast.hpp>

#include <prl/distributed/pairwise_averaging.hpp>
#include <prl/distributed/chord_random_host.hpp>
#include <prl/distributed/address_functions.hpp>

int main(int argc, char** argv) {
  using namespace std;
  using namespace prl;
  if (argc < 6) { 
    cerr << "Usage: test_chord_pairwise_averaging value host-name port chord-port niters" << endl;
    return EXIT_FAILURE;
  }
  
  double value         = boost::lexical_cast<double>(argv[1]);
  std::string hostname = argv[2];
  unsigned short port  = boost::lexical_cast<unsigned short>(argv[3]);
  unsigned short cport = boost::lexical_cast<unsigned short>(argv[4]);
  size_t niters        = boost::lexical_cast<size_t>(argv[5]);

  std::string hostip = address_string(hostname);
  size_t hash = boost::hash_value(std::make_pair(hostname, port));
  srandom(hash);

  chord_random_host hostfn(hostip, cport, port);
  pairwise_averaging<double> averaging(value, hostip, port, &hostfn);
  
  for(size_t i = 0; i < niters; i++) {
    averaging.iterate();
    averaging.lock();
    cout << averaging.value() << " ";
    averaging.unlock();
    cout.flush();
    boost::this_thread::sleep(boost::posix_time::seconds(1));      
  }
  cout << endl;
}

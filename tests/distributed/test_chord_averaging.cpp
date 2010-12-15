#include <iostream>
#include <cstdlib>

#include <boost/lexical_cast.hpp>

#include <prl/distributed/pushsum_averaging.hpp>
#include <prl/distributed/asynchronous_gossip.hpp>
#include <prl/distributed/synchronous_gossip.hpp>
#include <prl/distributed/chord_random_host.hpp>
#include <prl/distributed/address_functions.hpp>

int main(int argc, char** argv) {
  using namespace std;
  using namespace prl;
  if (argc < 7) { 
    cerr << "Usage: test_chord_averaging value host-name port chord-port niters algorithm [arguments]" << endl;
    return EXIT_FAILURE;
  }
  
  double value          = boost::lexical_cast<double>(argv[1]);
  std::string hostname  = argv[2];
  unsigned short port   = boost::lexical_cast<unsigned short>(argv[3]);
  unsigned short cport  = boost::lexical_cast<unsigned short>(argv[4]);
  size_t niters         = boost::lexical_cast<size_t>(argv[5]);
  std::string algorithm = argv[6];

  std::string hostip = address_string(hostname);
  size_t hash = boost::hash_value(std::make_pair(hostname, port));
  srandom(hash);

  chord_random_host hostfn(hostip, cport, port);
  distributed_averaging<double>* averaging;
  
  if (algorithm == "pushsum") {
    bool symmetric = argc < 8 ? false : boost::lexical_cast<bool>(argv[7]);
    averaging = new pushsum_averaging<double>(value, hostip, port, &hostfn,
					      symmetric, 10);
  } else if (algorithm == "async") {
    averaging = new asynchronous_gossip<double>(value, hostip, port, &hostfn, 
						false, 10);
  } else if (algorithm == "sync") {
    averaging = new synchronous_gossip<double>(value, hostip, port, &hostfn);
  } else {
    cerr << "Wrong algorithm" << endl;
    return EXIT_FAILURE;
  }

  for(size_t i = 0; i < niters; i++) {
    averaging->iterate();
    averaging->lock();
    cout << averaging->value() << endl; // " ";
    averaging->unlock();
    cout.flush();
    averaging->sleep(1);
  }
  cout << endl;
}

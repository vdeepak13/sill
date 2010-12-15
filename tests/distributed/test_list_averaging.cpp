#include <iostream>
#include <cstdlib>

#include <boost/functional/hash.hpp>
#include <boost/lexical_cast.hpp>

#include <prl/distributed/pushsum_averaging.hpp>
#include <prl/distributed/asynchronous_gossip.hpp>
#include <prl/distributed/synchronous_gossip.hpp>
#include <prl/distributed/synchronous_gossip_timed.hpp>
#include <prl/distributed/list_random_host.hpp>

int main(int argc, char** argv) {
  using namespace std;
  using namespace prl;
  if (argc != 7) { 
    cerr << "Usage: test_list_averaging value host-name port host-file niters algorithm" << endl;
    return EXIT_FAILURE;
  }
  
  double value         = boost::lexical_cast<double>(argv[1]);
  std::string hostname = argv[2];
  unsigned short port  = boost::lexical_cast<unsigned short>(argv[3]);
  list_random_host hostfn(argv[4]);
  size_t niters        = boost::lexical_cast<size_t>(argv[5]);
  std::string algorithm = argv[6];
  
  size_t hash = boost::hash_value(std::make_pair(hostname, port));
  srand(hash);

  distributed_averaging<double>* averaging;
  
  if (algorithm == "pushsum" || algorithm == "pushpull") {
    bool symmetric = (algorithm == "pushpull");
    averaging = new pushsum_averaging<double>(value, hostname, port, &hostfn,
					      symmetric, 10);
  } else if (algorithm == "async") {
    averaging = new asynchronous_gossip<double>(value, hostname, port, &hostfn, 
						false, 10);
  } else if (algorithm == "sync") {
    averaging = new synchronous_gossip<double>(value, hostname, port, &hostfn);
  } else if (algorithm == "synctimed") {
    averaging = new synchronous_gossip_timed<double>(value, hostname, port, 
						     &hostfn, 10);
  } else {
    cerr << "Wrong algorithm" << endl;
    return EXIT_FAILURE;
  }

  for(size_t i = 0; i < niters; i++) {
    averaging->iterate();
    averaging->lock();
    cout << averaging->value() << " ";
    averaging->unlock();
    cout.flush();
    averaging->sleep(1);      
  }
  cout << endl;
}

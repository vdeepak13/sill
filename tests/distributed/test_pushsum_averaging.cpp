#include <iostream>
#include <cstdlib>

#include <boost/functional/hash.hpp>
#include <boost/lexical_cast.hpp>

#include <prl/distributed/pushsum_averaging.hpp>
#include <prl/distributed/list_random_host.hpp>

int main(int argc, char** argv) {
  using namespace std;
  using namespace prl;
  if (argc < 6) { 
    cerr << "Usage: test_pushsum_averaging value host-name port host-file niters" << endl;
    return EXIT_FAILURE;
  }
  
  double value         = boost::lexical_cast<double>(argv[1]);
  std::string hostname = argv[2];
  unsigned short port  = boost::lexical_cast<unsigned short>(argv[3]);
  list_random_host hostfn(argv[4], port);
  size_t niters        = boost::lexical_cast<size_t>(argv[5]);
  size_t hash = boost::hash_value(std::make_pair(hostname, port));
  srand(hash);

  pushsum_averaging<double> averaging(value, hostname, port, &hostfn, false,
				      10);
  
  for(size_t i = 0; i < niters; i++) {
    averaging.iterate();
    averaging.lock();
    cout << averaging.value() << endl; //" ";
    averaging.unlock();
    cout.flush();
    averaging.sleep(1);
  }
  cout << endl;
}

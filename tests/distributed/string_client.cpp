#include <iostream>
#include <boost/asio.hpp>

#include <prl/serialization/serialize.hpp>

int main(int argc, char** argv) {
  using namespace std;
  using namespace prl;
  using boost::asio::ip::tcp;

  if (argc < 2) {
    cerr << "Usage: string_client host" << endl;
    return EXIT_FAILURE;
  }

  tcp::iostream stream(argv[1], "10000");
  assert(stream);
  iarchive iar(stream);
  std::string str;
  iar >> str;
  cout << "Received string of length " << str.size() << endl;

  return 0;
}

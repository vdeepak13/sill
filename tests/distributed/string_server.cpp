#include <boost/asio.hpp>
#include <boost/lexical_cast.hpp>

#include <prl/serialization/serialize.hpp>

int main(int argc, char** argv) {
  using namespace std;
  using namespace prl;
  using boost::asio::ip::tcp;

  if (argc < 2) {
    cerr << "Usage: string_server length" << endl;
    return EXIT_FAILURE;
  }

  size_t n = boost::lexical_cast<size_t>(argv[1]);

  std::string str(n, 1);

  boost::asio::io_service io_service;
  tcp::endpoint endpoint(tcp::v4(), 10000);
  tcp::acceptor acceptor(io_service, endpoint);

  while(1) {
    tcp::iostream stream;
    acceptor.accept(*stream.rdbuf());
    oarchive oar(stream);
    oar << str;
  }
}

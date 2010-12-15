// tests if a port is free

#include <iostream>

#include <boost/asio.hpp>
#include <boost/lexical_cast.hpp>

int main(int argc, char** argv) {

  using namespace std;
  using boost::asio::ip::tcp;
  
  if (argc < 2) {
    cerr << "Usage: check_port port" << endl;
    return 1;
  }
 
  unsigned short port = boost::lexical_cast<unsigned short>(argv[1]);

  boost::asio::io_service io_service;
  tcp::endpoint endpoint(tcp::v4(), port);
  try {
    tcp::acceptor acceptor(io_service, endpoint);
    cout << "Port " << port << " is available" << endl;
  } catch (std::exception& e) {
    cout << "Port " << port << " is not available" << endl;
  }
}

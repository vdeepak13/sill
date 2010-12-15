#include <iostream>

#include <boost/asio.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/thread/thread.hpp>

using boost::asio::ip::tcp;

size_t length;
char* data;

void session(tcp::iostream* stream) {
  *stream << length << std::endl;
  stream->write(data, length);
  delete stream;
}

int main(int argc, char** argv) {
  using namespace std;

  if (argc < 3) {
    cerr << "Usage: bandwidth_test_server port length" << endl;
    return -1;
  }

  unsigned short port = boost::lexical_cast<unsigned short>(argv[1]);
  length = boost::lexical_cast<size_t>(argv[2]);
  data = new char[length];

  boost::asio::io_service io_service;
  tcp::endpoint endpoint(tcp::v4(), port);
  tcp::acceptor acceptor(io_service, endpoint);

  while(1) {
    tcp::iostream* stream = new tcp::iostream;
    acceptor.accept(*stream->rdbuf());
    boost::thread thread(boost::bind(session, stream));
  }

}

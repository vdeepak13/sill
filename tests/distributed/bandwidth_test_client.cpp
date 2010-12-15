#include <iostream>

#include <boost/asio.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
//#include <boost/timer.hpp>

int main(int argc, char** argv) {
  using namespace std;
  using boost::asio::ip::tcp;
  using namespace boost::posix_time;

  if (argc < 3) {
    cerr << "Usage: bandwidth_test_client host port" << endl;
    return -1;
  }

  string hostname = argv[1];
  string port = argv[2];
  
  //cerr << "Connecting" << endl;
  tcp::iostream stream(hostname, port);
  assert(stream);

  string line;
  getline(stream, line);
  size_t length = boost::lexical_cast<size_t>(line);
  //cerr << "Allocating " << length << " bytes" << endl;
  char* data = new char[length];
  //cerr << "Reading " << length << " bytes" << endl;
  ptime start = microsec_clock::local_time();
  stream.read(data, length);
  ptime stop = microsec_clock::local_time();
  assert(stream);
  double duration = (stop-start).total_microseconds() / 1e6;
  cout << (length / duration) << endl;
  //cerr << "Read in " << duration << " seconds" << endl;
  
}

#include <fstream>
#include <sstream>
#include <cstdlib>

#include <boost/lexical_cast.hpp>

#include <prl/distributed/list_random_host.hpp>
#include <prl/parsers/string_functions.hpp>

namespace prl {

  list_random_host::list_random_host(const char* filename) { 
    std::ifstream in(filename);
    assert(in);
    std::string line;
    while(getline(in, line)) {
      tokenizer tok(line, " \t");
      std::string host = tok.next_token();
      unsigned short port = 
        boost::lexical_cast<unsigned short>(tok.next_token());
      assert(!tok.has_token());
      hosts.push_back(std::make_pair(host, port));
    }
  }

  list_random_host::list_random_host(const char* filename, unsigned short port){
    std::ifstream in(filename);
    assert(in);
    std::string line;
    while(getline(in, line)) {
      hosts.push_back(std::make_pair(line, port));
    }
  }

  std::pair<std::string, unsigned short> list_random_host::operator()() const {
    return hosts[rand() % hosts.size()];
  }

}

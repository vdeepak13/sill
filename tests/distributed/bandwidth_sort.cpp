#include <fstream>
#include <algorithm>
#include <utility>
#include <string>
#include <iostream>
#include <vector>

#include <boost/lexical_cast.hpp>

struct compare {
  typedef std::pair<double, const char*> value_type;

  bool operator()(const value_type& a, const value_type& b) const {
    return a.first > b.first;
  }
};

int main(int argc, char** argv) {
  using namespace std;
  if (argc < 2) {
    cerr << "Usage: hostfile1 ..." << endl;
    return 1;
  }
  
  std::vector< std::pair<double, const char*> > values;

  for(size_t i = 1; i < size_t(argc); i++) {
    ifstream in(argv[i]);
    std::string line;
    getline(in, line);
    try {
      double bw = boost::lexical_cast<double>(line);
      values.push_back(std::make_pair(bw, argv[i]));
    } catch(std::exception& e) {
      cerr << "Ignoring " << argv[i] << ": " << line << endl;
    }
  }

  std::sort(values.begin(), values.end(), compare());
  
  for(size_t i = 0; i < values.size(); i++) {
    cout << /* values[i].first << " " << */ values[i].second << endl;
  }
   
  
}

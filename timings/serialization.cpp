#include <iostream>
#include <fstream>
#include <stdexcept>

#include <boost/timer.hpp>
#include <boost/lexical_cast.hpp>

#include <sill/serialization/serialize.hpp>
#include <sill/serialization/vector.hpp>
#include <sill/serialization/map.hpp>
#include <sill/serialization/list.hpp>
#include <sill/serialization/set.hpp>

using namespace sill;

int main(int argc, char** argv) {
  using namespace std;

  if (argc != 3) {
    cerr << "usage: ./serialization [iterations to run each test for]"
         << " [buffer size]"
         << endl;
    return 1;
  }

  size_t n = boost::lexical_cast<size_t>(argv[1]);
  //  size_t BUFFER_SIZE = boost::lexical_cast<size_t>(argv[2]);

  {
    ofstream out("serialization.tmp");
//    char* buf = new char[BUFFER_SIZE];
//    out.rdbuf()->pubsetbuf(buf, BUFFER_SIZE);
    oarchive oar(out);
    const size_t NUM_ELEMENTS = 1000;
    std::map<size_t,size_t> mymap;
    for (size_t i = 0; i < NUM_ELEMENTS; ++i)
      mymap[i] = i+1;
    boost::timer t;
    for(size_t i = 0; i < n; i++)
      oar << mymap;
    out.close();
    double elapsed = t.elapsed();
    cout << "Serialized map with " << NUM_ELEMENTS << " elements " << n
         << " times in " << elapsed << " seconds" << endl;
//    delete [] buf;
//    buf = NULL;

    ifstream in("serialization.tmp");
    iarchive iar(in);
    mymap.clear();
    t.restart();
    for (size_t i = 0; i < n; ++i)
      iar >> mymap;
    in.close();
    elapsed = t.elapsed();
    cout << "Deserialized map with " << NUM_ELEMENTS << " elements " << n
         << " times in " << elapsed << " seconds" << endl;
  }

  {
    ofstream out("serialization.tmp");
//    char* buf = new char[BUFFER_SIZE];
//    out.rdbuf()->pubsetbuf(buf, BUFFER_SIZE);
    oarchive oar(out);
    const size_t NUM_ELEMENTS = 1000;
    std::vector<std::pair<size_t,size_t> > myvec;
    for (size_t i = 0; i < NUM_ELEMENTS; ++i)
      myvec.push_back(std::make_pair(i, i+1));
    boost::timer t;
    for(size_t i = 0; i < n; i++)
      oar << myvec;
    out.close();
    double elapsed = t.elapsed();
    cout << "Serialized vec of pairs with " << NUM_ELEMENTS << " elements " << n
         << " times in " << elapsed << " seconds" << endl;
//    delete [] buf;
//    buf = NULL;

    ifstream in("serialization.tmp");
    iarchive iar(in);
    myvec.clear();
    t.restart();
    for (size_t i = 0; i < n; ++i)
      iar >> myvec;
    in.close();
    elapsed = t.elapsed();
    cout << "Deserialized vec with " << NUM_ELEMENTS << " elements " << n
         << " times in " << elapsed << " seconds" << endl;
  }

}

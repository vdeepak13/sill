#include <iostream>
#include <fstream>
#include <stdexcept>

#include <boost/timer.hpp>
#include <boost/lexical_cast.hpp>

class oarchive{
public:
  std::ostream* o;
  size_t bytes_;

  oarchive(std::ostream& os)
    : o(&os), bytes_() {}

  ~oarchive() { }

  size_t bytes() { 
    return bytes_;
  }
};

inline void check(std::ostream* out) {
  if (out->fail()) {
    throw std::runtime_error("oarchive: Stream operation failed!");
  }
}

oarchive& serialize_plain(oarchive& a, const double i) {
  a.o->write(reinterpret_cast<const char*>(&i), sizeof(double));
  check(a.o);
  return a;
}

inline oarchive& serialize_inlined(oarchive& a, const double i) {
  a.o->write(reinterpret_cast<const char*>(&i), sizeof(double));
  check(a.o);
  return a;
}

inline oarchive& serialize_counting(oarchive& a, const double i) {
  a.o->write(reinterpret_cast<const char*>(&i), sizeof(double));
  a.bytes_ += sizeof(double);
  check(a.o);
  return a;
}


int main(int argc, char** argv) {
  using namespace std;

  size_t n = boost::lexical_cast<size_t>(argv[1]);

  {
    boost::timer t;
    ofstream out("/dev/null");
    oarchive oar(out);
    for(size_t i = 0; i < n; i++)
      serialize_plain(oar, 0.0);
    cout << "Plain: " << t.elapsed() << " s" << endl;
  }

  {
    boost::timer t;
    ofstream out("/dev/null");
    oarchive oar(out);
    for(size_t i = 0; i < n; i++)
      serialize_inlined(oar, 0.0);
    cout << "Inlined: " << t.elapsed() << " s" << endl;
  }

  {
    boost::timer t;
    ofstream out("/dev/null");
    oarchive oar(out);
    for(size_t i = 0; i < n; i++)
      serialize_counting(oar, 0.0);
    cout << "Counting bytes: " << t.elapsed() << " s" << endl;
    cout << oar.bytes() << endl;
  }

}

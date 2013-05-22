#include <iostream>
#include <fstream>
#include <string>
#include <cassert>

#include <boost/lexical_cast.hpp>

#include <sill/map.hpp>
#include <sill/range/io.hpp>

int main(int argc, char** argv) {
  using namespace sill;
  using namespace std;

  typedef sill::map<int, string> itoa_map;

  itoa_map s1, s2, s3, letters;

  for (int i = 1; i < 5; i++) {
    s1[i] = boost::lexical_cast<string>(i);
  }
  for (int i = 3; i < 7; i++) {
    s2[i] = boost::lexical_cast<string>(i);
  }
  for (int i=1; i<10; i++) {
    letters[i] = boost::lexical_cast<string>(char('@' + i));
  }

  s3 = s1 + s2;

  cout << "S1: " << s1 << endl;

  cout << "S2: " << s2 << endl;

  cout << "S1 && S2: " << s1.intersect(s2) << endl;

  cout << "S1 || S2: " << (s1 + s2) << endl;

  cout << "S2 - S1: " << (s2 - s1) << endl;

  cout << "values=" << (s1+s2).values(s1.keys()) << endl;

  //cout << "values=" << (s1-s2).values(s1.keys(), string("empty")) << endl;

  cout << "letters=" << letters << endl;

  cout << "s3.rekey(letters)" << s3.rekey(letters) << endl;
  
  std::ofstream fout("test.bin");
  oarchive a(fout);
  a << s1;
  a << s2;
  a << s3;
  fout.close();
  
  ifstream fin("test.bin");
  iarchive b(fin);
  b >> s1;
  b >> s2;
  b >> s3;
  fin.close();
  
  cout << "S1: " << s1 << endl;
  cout << "S2: " << s2 << endl;
  
  // Return success.
  return EXIT_SUCCESS;
}

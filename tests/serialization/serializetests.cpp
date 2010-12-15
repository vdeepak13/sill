#include <fstream>
#include <cassert>
#include <vector>
#include <map>
#include <string>
#include <cstring>

#include <prl/serialization/serialize.hpp>
#include <prl/serialization/vector.hpp>
#include <prl/serialization/map.hpp>
#include <prl/serialization/list.hpp>
#include <prl/serialization/set.hpp>

using namespace std;
using namespace prl;

// using namespace prl;
// Look for the class TestClass() to see the most interesting tutorial on how to 
// use the serializer
void test_basic_datatype() {
  char t1 = 'z';
  bool t2 = true;  
  int t3 = 10;
  int t4 = 18345;
  long t5 = 30921233;
  long long t6 = (long long)(t5)*100;
  float t7 = 10.35;
  double t8 = 3.14156;
  const char *t9 = "hello world";
  const char * t10 = "blue";
  
  char r1;
  bool r2;  
  int r3;
  int r4;
  long r5;
  long long r6;
  float r7;
  double r8;
  char r9[100];
  char r10[10];
  
  // serialize t1-10
  ofstream f;
  f.open("test.bin",fstream::binary);
  oarchive a(f);
  a << t1 << t2 << t3 << t4 << t5 << t6 << t7 << t8;
  prl::serialize(a, t9, strlen(t9) + 1);
  prl::serialize(a, t10, strlen(t10) + 1);
  f.close();
  
  // deserialize into r1-10
  ifstream g;
  g.open("test.bin",fstream::binary);
  iarchive b(g);
  b >> r1 >> r2 >> r3 >> r4 >> r5 >> r6 >> r7 >> r8;
  prl::deserialize(b, &r9, strlen(t9) + 1);
  prl::deserialize(b, r10, strlen(t10) + 1);
  g.close();
  
  assert(t1 == r1);
  assert(t2 == r2);
  assert(t3 == r3);
  assert(t4 == r4);
  assert(t5 == r5);
  assert(t6 == r6);
  assert(t7 == r7);
  assert(t8 == r8);
  assert(strcmp(t9, r9) == 0);
  assert(strcmp(t10, r10) == 0);
}

void test_vectors() {
  vector<int> v;
  for (int i = 0;i< 10; ++i) {
    v.push_back(i);
  }
  ofstream f;
  f.open("test.bin",fstream::binary);
  oarchive a(f);
  a << v;
  f.close();
  
  vector<int> w;
  ifstream g;
  iarchive b(g);
  g.open("test.bin",fstream::binary);
  b >> w;
  g.close();
  
  for (int i = 0;i< 10; ++i) {
    assert(v[i]==w[i]);
  }
}

struct A{
  int z;
  void save(oarchive &a) const { 
    a << z;
  } 
  void load(iarchive a) { 
    a >> z;
  } 
};

class TestClass{
public:
  int i;
  int j;
  vector<int> k;
  A l;
  void save(oarchive &a) const { 
    a << i << j << k << l;
  } 
  void load(iarchive &a) { 
    a >> i >> j >> k >> l;
  } 
};

void test_classes() {
  // create a test class
  TestClass t;
  t.i=10;
  t.j=20;
  t.k.push_back(30);
  
  //serialize
  ofstream f;
  f.open("test.bin",fstream::binary);
  oarchive a(f);
  a << t;
  f.close();
  //deserialize into t2
  TestClass t2;
  ifstream g;
  g.open("test.bin",fstream::binary);
  iarchive b(g);
  b >> t2;
  g.close();
  // check
  assert(t.i == t2.i);
  assert(t.j == t2.j);
  assert(t.k.size() == t2.k.size());
  assert(t.k[0] == t2.k[0]);
}

void test_vector_of_classes() {
  // create a vector of test classes
  vector<TestClass> vt;
  vt.resize(10);
  for (int i=0;i<10;i++) {
    vt[i].i=i;
    vt[i].j=i*21;
    vt[i].k.resize(10);
    vt[i].k[i]=i*51;
  }
  
  //serialize
  ofstream f;
  f.open("test.bin",fstream::binary);
  oarchive a(f);
  a << vt;
  f.close();
  
  //deserialize into vt2
  vector<TestClass> vt2;
  ifstream g;
  g.open("test.bin",fstream::binary);
  iarchive b(g);
  b >> vt2;
  g.close();
  // check
  assert(vt.size() == vt2.size());  
  for (size_t i=0;i<10;i++) {
    assert(vt[i].i == vt2[i].i);
    assert(vt[i].j == vt2[i].j);
    assert(vt[i].k.size() == vt2[i].k.size());
    for (size_t j = 0; j < vt[i].k.size(); ++j) {
      assert(vt[i].k[j] == vt2[i].k[j]);
    }
  }
}

void test_vector_of_strings() {
  string x = "Hello world";
  string y = "This is a test";
  vector<string> v;
  v.push_back(x); v.push_back(y);
  
  ofstream f;
  f.open("test.bin",fstream::binary);
  oarchive a(f);
  a << v;
  f.close();
  
  //deserialize into vt2
  vector<string> v2;
  ifstream g;
  g.open("test.bin",fstream::binary);
  iarchive b(g);
  b >> v2;
  g.close();
  
  assert(v[0] == v2[0]);
  assert(v[1] == v2[1]);
}

void test_map() {
  map<string,int> v;
  v["one"] = 1;
  v["two"] = 2;
  v["three"] = 3;
  
  ofstream f;
  f.open("test.bin",fstream::binary);
  oarchive a(f);
  a << v;
  f.close();
  
  //deserialize into vt2
  map<string,int> v2;
  ifstream g;
  g.open("test.bin",fstream::binary);
  iarchive b(g);
  b >> v2;
  g.close();
  
  assert(v["one"] == v2["one"]);
  assert(v["two"] == v2["two"]);
  assert(v["three"] == v2["three"]);
}

int main(int argc, char** argv) {
  test_basic_datatype();
  test_vectors();
  test_classes();
  test_vector_of_classes();
  test_vector_of_strings();
  test_map();
  std::cout << "All Ok\n";
}

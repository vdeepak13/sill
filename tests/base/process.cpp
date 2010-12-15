#include <iostream>
#include <fstream>

#include <prl/base/universe.hpp>
#include <prl/base/timed_process.hpp>
#include <prl/serialization/serialize.hpp>

using namespace std;
using namespace prl;

int main()
{
  // serialize 
  {
  universe u;
  finite_timed_process* p = new finite_timed_process("p", 4);
  finite_timed_process* q = new finite_timed_process("q", 2);
  cout << p << endl;
  cout << p->current() << endl;
  cout << p->next() << endl;
  cout << p->at(1) << endl;
  cout << p->at(2) << endl;
  assert(p->at(1) == p->at(1));
  u.add(p);
  u.add(q);
  
  ofstream fout("test.bin");
  oarchive a(fout);
  a << u;
  fout.close();
  
  std::set<finite_timed_process*> procs = make_domain(p, q);
  cout << procs << endl;
  cout << variables(procs, current_step) << endl;
  cout << variables(procs, 1) << endl;
  cout << processes<finite_timed_process>(variables(procs, 1)) << endl;
  } 
   
  // deserialize 
  {
    ifstream fin("test.bin");
    iarchive  i(fin);
    universe u2;
    i >> u2;
    finite_timed_process* p 
      = dynamic_cast<finite_timed_process*>(u2.proc_from_id(0));
    finite_timed_process* q 
      = dynamic_cast<finite_timed_process*>(u2.proc_from_id(1));

    cout << p << endl;
    cout << p->current() << endl;
    cout << p->next() << endl;
    cout << p->at(1) << endl;
    cout << p->at(2) << endl;
    assert(p->at(1) == p->at(1));
    
    std::set<finite_timed_process*> procs = make_domain(p, q);
    cout << procs << endl;
    cout << variables(procs, current_step) << endl;
    cout << variables(procs, 1) << endl;
    cout << processes<finite_timed_process>(variables(procs, 1)) << endl;
    
  }
}

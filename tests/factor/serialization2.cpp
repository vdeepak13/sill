#include <fstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/export.hpp>

#include <sill/factor/canonical_gaussian.hpp>
#include <sill/base/named_universe.hpp>
#include <sill/math/linear_algebra.hpp>

#include <sill/process.hpp>

int main()
{
  using namespace sill;
  using namespace std;

  named_universe u;
  vector_timed_process* xp = u.add_process(new vector_timed_process("x", 1));
  vector_timed_process* yp = u.add_process(new vector_timed_process("y", 2));
  vector_variable* x = xp->current();
  vector_variable* y = yp->current();

  canonical_gaussian cg(make_domain(x, y), identity(3), "1 2 3");

//   std::ofstream ofs("test.txt");
//   boost::archive::text_oarchive oa(ofs);
//   oa & cg;
//   ofs.close();

  canonical_gaussian cg2;
  std::ifstream ifs("test.txt");
  boost::archive::text_iarchive ia(ifs);
  ia & cg2;

  cout << cg2 << endl;
  cout << cg2.arguments() << endl;  
  cout << (cg2.arguments() == cg.arguments()) << endl;
  
  cg2.subst_args(u);
  cout << cg2 << endl;
  cout << cg2.arguments() << endl;
  cout << (cg2.arguments() == cg.arguments()) << endl;
}

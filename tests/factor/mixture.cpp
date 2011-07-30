#include <iostream>

#include <sill/factor/moment_gaussian.hpp>
#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/mixture.hpp>

int main() {
  using namespace sill;
  using namespace std;

  universe u;
  vector_var_vector v = u.new_vector_variables(2, 1);

  mixture_gaussian mix(2, vector_domain(v.begin(), v.end()));
  mix[0] = moment_gaussian(v, zeros(2), eye(2,2));
  mix[1] = moment_gaussian(v, ones(2), "2 1; 1 3");

  cout << mix << endl;

  mix.add_parameters(mix, 0.5);
  cout << mix << endl;
  
  mix.normalize();
  cout << mix << endl;

  // Test serialization.
  ofstream fout("test.bin",fstream::binary);
  oarchive oa(fout);
  oa << mix;
  fout.close();

  ifstream fin("test.bin",fstream::binary);
  iarchive ia(fin);
  ia.attach_universe(&u);
  mixture_gaussian mix2;
  ia >> mix2;
  fin.close();
  assert(mix == mix2);

  return 0;
}

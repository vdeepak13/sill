#include <vector>
#include <iostream>

#include <boost/random/mersenne_twister.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/random.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/learning/dataset/record_conversions.hpp>

#include <sill/macros_def.hpp>

int main() {
  using namespace sill;
  using namespace std;

  boost::mt19937 rng(time(NULL));
  universe u;
  
  finite_var_vector vars = u.new_finite_variables(2, 2);
  std::vector<double> values;
  values.push_back(1);
  values.push_back(2);
  values.push_back(3);
  values.push_back(4);

  table_factor f(vars, values);
  cout << f << endl;
  cout << pow(f, 2) << endl;

  // Test unrolling and rolling up factors.
  finite_var_vector vars3 = u.new_finite_variables(3, 2);
  table_factor f_rolled = random_range_discrete_factor<table_factor>
    (make_domain(vars3), rng, 0, 1);
  finite_variable* f_unrolled_v = NULL;
  table_factor f_unrolled;
  boost::tie(f_unrolled_v, f_unrolled) = f_rolled.unroll(u);
  table_factor f_rolled_again = f_unrolled.roll_up(f_rolled.arg_list());
  if (f_rolled != f_rolled_again) {
    cerr << "In test of unrolling and rolling up factors, "
         << " rolling up did not reverse unrolling.\n"
         << "Original (rolled) factor: " << f_rolled
         << "\nUnrolled factor: " << f_unrolled
         << "\nRolled-up factor: " << f_rolled_again << endl;
    return 1;
  } else {
    cout << "Table factor unrolling and rolling was successful.\n" << endl;
  }

  // Test different methods of restricting factors.
  finite_assignment fa;
  fa[vars3[0]] = 1;
  fa[vars3[1]] = 0;
  finite_var_vector fa_vars;
  fa_vars.push_back(vars3[0]);
  fa_vars.push_back(vars3[1]);
  std::vector<size_t> fr_data;
  finite_assignment2vector(fa, fa_vars, fr_data);
  finite_record fr(fa_vars);
  fr.set_finite_val(fr_data);

  table_factor f1(f_rolled);
  cout << "Testing restricting factors\n"
       << "========================================================\n\n";
  cout << "Factor f1:\n" << f1 << endl;
  table_factor new_f1a;
  f1.restrict(fa, new_f1a);
  cout << "f1 restricted by assignment " << fa << ":\n"
       << new_f1a << endl;
  table_factor new_f1b;
  f1.restrict(fr, new_f1b);
  assert(new_f1a == new_f1b);

  f1.restrict(fa, make_domain(vars3[1]), new_f1a);
  cout << "f1 restricted by assignment " << fa << " limited to variables "
       << make_domain(vars3[1]) << ":\n"
       << new_f1a << endl;
  f1.restrict(fr, make_domain(vars3[1]), new_f1b);
  assert(new_f1a == new_f1b);

  return 0;
}

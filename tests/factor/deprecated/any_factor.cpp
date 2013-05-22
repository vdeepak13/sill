#include <iostream>
#include <vector>

#include <boost/lexical_cast.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <sill/factor/any_factor.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/factor/random/random.hpp>
#include <sill/inference/junction_tree_inference.hpp>
#include <sill/stl_io.hpp>

#include <sill/range/algorithm.hpp>

boost::mt19937 rng;

int main(int argc, char** argv)
{
  using namespace sill;
  using namespace std;

  /*
  typedef table_factor< dense_table<double> > table_factor;
  typedef any_factor<double> any_factor;
  */

  /*
  typedef any_factor<double> factor_type;
  factor_type f = tablef(1);
  factor_type f1= tablef(1);
  factor_type g = constant_factor(2);
  factor_type h = f*g;
  cout << f << endl;
  cout << g << endl;
  cout << h << endl;
  cout << (f == f1) << endl;
  cout << (f == g) << endl;
  */

  size_t m = (argc > 1) ? boost::lexical_cast<size_t>(argv[1]) : 2;
  size_t n = (argc > 2) ? boost::lexical_cast<size_t>(argv[2]) : 100;
  size_t k = (m * n) / 2;

  universe u;
  finite_var_vector v = u.new_finite_variables(k, 2);
  std::vector< tablef > factors(n);

  // Generate n random table factors, each with <=m variables
  for(size_t i = 0; i < n; i++) {
    finite_domain d;
    for(size_t j = 0; j < m; j++) d.insert(v[rng() % k]);
    factors[i] = random_discrete_factor< tablef >(d, rng);
  }

  if (n < 10) {
    cout << "Random factors:" << endl;
    cout << v << endl;
  }

  // Perform junction tree inference using the table factors
  shafer_shenoy<tablef> ss_table(factors);
  cout << "Tree width of ss_table: " << ss_table.tree_width() << endl;
  ss_table.calibrate();
  ss_table.normalize();
  std::vector< tablef > table_beliefs = ss_table.clique_beliefs();
  if (n < 10) cout << table_beliefs << endl;

  // Perform junction tree inference using polymorphic factors
  shafer_shenoy<any_factor> ss_poly(factors);
  cout << "Tree width of ss_poly: " << ss_table.tree_width() << endl;
  ss_poly.calibrate();
  ss_poly.normalize();
  std::vector<any_factor> poly_beliefs = ss_poly.clique_beliefs();
  if (n < 10) cout << poly_beliefs << endl;

  assert(norm_inf(poly_beliefs[0], poly_beliefs[0]) < 1e-10);

  assert(table_beliefs[0] == poly_beliefs[0]);
  //assert(!(poly_beliefs[0] < poly_beliefs[0]));

  // Compare the computed beliefs
  assert(poly_beliefs.size() == table_beliefs.size());
  assert(sill::equal(poly_beliefs, table_beliefs));
}

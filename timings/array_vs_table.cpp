#include <sill/base/universe.hpp>
#include <sill/factor/probability_array.hpp>
#include <sill/factor/probability_table.hpp>

#include <boost/timer.hpp>

using namespace sill;
using namespace std;

typedef probability_array<double> arrayf;
typedef probability_table<double> tablef;

void test_join(size_t nit,
               const finite_var_vector& doma,
               const finite_var_vector& domb) {
  for (finite_variable* v : doma) cout << v->name() << " ";
  cout << "* ";
  for (finite_variable* v : domb) cout << v->name() << " ";
  cout << ": ";
  boost::timer t;

  arrayf fa(doma);
  arrayf ga(domb);
  t.restart();
  for (size_t i = 0; i < nit; ++i) {
    arrayf ha = fa * ga;
  }
  cout << t.elapsed() << " ";

  tablef ft(doma);
  tablef gt(domb);
  t.restart();
  for (size_t i = 0; i < nit; ++i) {
    tablef ht = ft * gt;
  }
  cout << t.elapsed() << endl;
}

void test_join_inplace(size_t nit,
                       const finite_var_vector& doma,
                       const finite_var_vector& domb) {
  for (finite_variable* v : doma) cout << v->name() << " ";
  cout << "*= ";
  for (finite_variable* v : domb) cout << v->name() << " ";
  cout << ": ";
  boost::timer t;

  arrayf fa(doma);
  arrayf ga(domb);
  t.restart();
  for (size_t i = 0; i < nit; ++i) {
    fa *= ga;
  }
  cout << t.elapsed() << " ";

  tablef ft(doma);
  tablef gt(domb);
  t.restart();
  for (size_t i = 0; i < nit; ++i) {
    ft *= gt;
  }
  cout << t.elapsed() << endl;
}


void test_aggregate(size_t nit,
                    const finite_var_vector& doma,
                    const finite_var_vector& domb) {
  for (finite_variable* v : doma) cout << v->name() << " ";
  cout << "-> ";
  for (finite_variable* v : domb) cout << v->name() << " ";
  cout << ": ";
  boost::timer t;

  arrayf fa(doma);
  t.restart();
  for (size_t i = 0; i < nit; ++i) {
    arrayf ga = fa.marginal(domb);
  }
  cout << t.elapsed() << " ";

  tablef ft(doma);
  t.restart();
  for (size_t i = 0; i < nit; ++i) {
    tablef gt = ft.marginal(make_domain(domb));
  }
  cout << t.elapsed() << endl;
}

int main(int argc, char** argv) {
  size_t dim = argc > 1 ? atoi(argv[1]) : 100;
  size_t nit = argc > 2 ? atoi(argv[2]) : 1000;

  universe u;
  finite_variable* x = u.new_finite_variable("x", dim);
  finite_variable* y = u.new_finite_variable("y", dim);

  test_join(nit, {x}, {x});
  test_join(nit, {x}, {y});
  test_join(nit, {x}, {x, y});
  test_join(nit, {x}, {y, x});
  test_join(nit, {x, y}, {});
  test_join(nit, {x, y}, {x});
  test_join(nit, {x, y}, {y});
  test_join(nit, {x, y}, {x, y});
  test_join(nit, {x, y}, {y, x});

  cout << endl;

  test_join_inplace(nit, {x}, {x});
  test_join_inplace(nit, {x, y}, {});
  test_join_inplace(nit, {x, y}, {x});
  test_join_inplace(nit, {x, y}, {y});
  test_join_inplace(nit, {x, y}, {x, y});
  test_join_inplace(nit, {x, y}, {y, x});

  cout << endl;

  test_aggregate(nit, {x, y}, {x});
  test_aggregate(nit, {x, y}, {y});
  test_aggregate(nit, {x, y}, {});

  cout << endl;
}

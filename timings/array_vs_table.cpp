#include <sill/base/universe.hpp>
#include <sill/factor/canonical_array.hpp>
#include <sill/factor/canonical_table.hpp>
#include <sill/factor/probability_array.hpp>
#include <sill/factor/probability_table.hpp>

#include <boost/timer.hpp>

using namespace sill;
using namespace std;

template <typename Array1, typename Array2, typename ArrayR, typename Table>
void test_join(size_t n,
               const finite_var_vector& doma,
               const finite_var_vector& domb) {
  for (finite_variable* v : doma) cout << v->name() << " ";
  cout << "* ";
  for (finite_variable* v : domb) cout << v->name() << " ";
  cout << ": ";
  boost::timer t;

  Array1 fa(doma);
  Array2 ga(domb);
  t.restart();
  for (size_t i = 0; i < n; ++i) {
    ArrayR ha = fa * ga;
  }
  cout << t.elapsed() << " ";

  Table ft(doma);
  Table gt(domb);
  t.restart();
  for (size_t i = 0; i < n; ++i) {
    Table ht = ft * gt;
  }
  cout << t.elapsed() << endl;
}

template <typename Array1, typename Array2, typename Table>
void test_join_inplace(size_t n,
                       const finite_var_vector& doma,
                       const finite_var_vector& domb) {
  for (finite_variable* v : doma) cout << v->name() << " ";
  cout << "*= ";
  for (finite_variable* v : domb) cout << v->name() << " ";
  cout << ": ";
  boost::timer t;

  Array1 fa(doma);
  Array2 ga(domb);
  t.restart();
  for (size_t i = 0; i < n; ++i) {
    fa *= ga;
  }
  cout << t.elapsed() << " ";

  Table ft(doma);
  Table gt(domb);
  t.restart();
  for (size_t i = 0; i < n; ++i) {
    ft *= gt;
  }
  cout << t.elapsed() << endl;
}


template <typename Array1, typename ArrayR, typename Table>
void test_aggregate(size_t n,
                    const finite_var_vector& doma,
                    const finite_var_vector& domb) {
  for (finite_variable* v : doma) cout << v->name() << " ";
  cout << "-> ";
  for (finite_variable* v : domb) cout << v->name() << " ";
  cout << ": ";
  boost::timer t;

  Array1 fa(doma);
  t.restart();
  for (size_t i = 0; i < n; ++i) {
    ArrayR ga = fa.marginal(domb);
  }
  cout << t.elapsed() << " ";

  Table ft(doma);
  t.restart();
  for (size_t i = 0; i < n; ++i) {
    Table gt = ft.marginal(make_domain(domb));
  }
  cout << t.elapsed() << endl;
}

template <typename Array1, typename Array2, typename Table>
void test_all(size_t d, size_t n) {
  universe u;
  finite_variable* x = u.new_finite_variable("x", d);
  finite_variable* y = u.new_finite_variable("y", d);

  test_join<Array1, Array1, Array1, Table>(n, {x}, {x});
  test_join<Array1, Array2, Array2, Table>(n, {x}, {x, y});
  test_join<Array1, Array2, Array2, Table>(n, {x}, {y, x});
  test_join<Array2, Array1, Array2, Table>(n, {x, y}, {x});
  test_join<Array2, Array1, Array2, Table>(n, {x, y}, {y});
  test_join<Array2, Array2, Array2, Table>(n, {x, y}, {x, y});
  test_join<Array2, Array2, Array2, Table>(n, {x, y}, {y, x});

  cout << endl;

  test_join_inplace<Array1, Array1, Table>(n, {x}, {x});
  test_join_inplace<Array2, Array1, Table>(n, {x, y}, {x});
  test_join_inplace<Array2, Array1, Table>(n, {x, y}, {y});
  test_join_inplace<Array2, Array2, Table>(n, {x, y}, {x, y});
  test_join_inplace<Array2, Array2, Table>(n, {x, y}, {y, x});

  cout << endl;

  test_aggregate<Array2, Array1, Table>(n, {x, y}, {x});
  test_aggregate<Array2, Array1, Table>(n, {x, y}, {y});

  cout << endl;
}

typedef probability_array<double, 1> pa1_type;
typedef probability_array<double, 2> pa2_type;
typedef canonical_array<double, 1> ca1_type;
typedef canonical_array<double, 2> ca2_type;


int main(int argc, char** argv) {
  size_t d = argc > 1 ? atoi(argv[1]) : 100;
  size_t n = argc > 2 ? atoi(argv[2]) : 1000;

  cout << "Probability factors:" << endl;
  cout << "--------------------" << endl;
  test_all<pa1_type, pa2_type, probability_table<double> >(d, n);
 
  cout << "Canonical factors:" << endl;
  cout << "------------------" << endl;
  test_all<ca1_type, ca2_type, canonical_table<double> >(d, n);

  return 0;
}


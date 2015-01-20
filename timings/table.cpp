#include <cassert>
#include <iostream>
#include <functional>

#include <boost/timer.hpp>

#include <sill/datastructure/dense_table.hpp>
#include <sill/datastructure/table.hpp>
#include <sill/functional.hpp>
#include <sill/range/io.hpp>

#include <sill/macros_def.hpp>

using namespace sill;
using namespace std;

typedef dense_table<double> double_table_old;
typedef table<double> double_table;

void time_join_variable() {
  cout << "\nJoin for various sizes (in MFLOPs);" << endl;
  boost::timer t;

  const size_t p = 4;
  const size_t d = 10;
  size_t n = 10 * std::pow(3, d) + 1;

  for (size_t k = 1; k <= d; ++k) {
    double_table_old x0({p});
    double_table_old y0({p});
    double_table_old r0(finite_index(k, p));

    double_table x({p});
    double_table y({p});
    double_table r(finite_index(k, p));

    finite_index x_map(1, 0);
    finite_index y_map(1, 0);
    std::plus<double> op;

    cout << k << " ";
    
    t.restart();
    for (size_t i = 0; i < n / 10; ++i) {
      r0.join(x0, y0, x_map, y_map, op);
    }
    cout << r0.size() * (n / 10) / t.elapsed() / 1e6 << " ";
    
    t.restart();
    for (size_t i = 0; i < n / 3; ++i) {
      table_join<double, double, std::plus<double> >(r, x, y, x_map, y_map, op).loop();
    }
    cout << r.size() * (n / 3) / t.elapsed() / 1e6 << " ";

    t.restart();
    for (size_t i = 0; i < n; ++i) {
      table_join<double, double, std::plus<double> >(r, x, y, x_map, y_map, op)();
    }
    cout << r.size() * n / t.elapsed() / 1e6 << std::endl;

    n /= 3;
  }
}

void time_join() {
  boost::timer t;
  cout << "\nJoin for fixed sizes:" << endl;

  const size_t p = 10;
  const size_t q = 8;
  const size_t r = 9;

  double e_start = 0;
  double f_start = 5;

  double_table_old e0({p, q});
  double_table_old f0({q, r});
  std::iota(e0.begin(), e0.end(), e_start);
  std::iota(f0.begin(), f0.end(), f_start);

  double_table e({p, q});
  double_table f({q, r});
  std::iota(e.begin(), e.end(), e_start);
  std::iota(f.begin(), f.end(), f_start);

  size_t n = 10000;

  finite_index e_map = {0, 1};
  finite_index f_map = {1, 2};
  std::plus<double> op;

  t.restart();
  for(size_t i = 0; i < n; ++i) {
    double_table_old g0({p, q, r});
    g0.join(e0, f0, e_map, f_map, op);
  }
  cout << "Performed " << n << " joins (old) in " << t.elapsed() << "s." << endl;

  t.restart();
  for (size_t i = 0; i < n; ++i) {
    double_table g({p, q, r});
    table_join<double, double, std::plus<double> >(g, e, f, e_map, f_map, op).loop();
  }
  cout << "Performed " << n << " joins (new) in " << t.elapsed() << "s." << endl;

  t.restart();
  for (size_t i = 0; i < n; ++i) {
    double_table g({p, q, r});
    table_join<double, double, std::plus<double> >(g, e, f, e_map, f_map, op)();
  }
  cout << "Performed " << n << " joins (inl) in " << t.elapsed() << "s." << endl;
}

void time_aggregate() {
  boost::timer t;
  cout << "\nAggregate for fixed sizes:" << endl;

  const size_t p = 10;
  const size_t q = 8;
  const size_t r = 9;

  double_table_old g0({p, q, r});
  std::iota(g0.begin(), g0.end(), 1);

  double_table g({p, q, r});
  std::iota(g.begin(), g.end(), 1);

  size_t n = 100000;
  finite_index h_map = {0, 2};
  std::plus<double> op;

  t.restart();
  for (size_t i = 0; i < n; ++i) {
    double_table_old h0({p, r});
    h0.aggregate(g0, h_map, op);
  }
  cout << "Performed " << n << " aggregates (old) in " << t.elapsed() << "s." << endl;

  t.restart();
  for (size_t i = 0; i < n; ++i) {
    double_table h({p, r});
    table_aggregate<double, double, std::plus<double> >(h, g, h_map, op).loop();
  }
  cout << "Performed " << n << " aggregates (new) in " << t.elapsed() << "s." << endl;

  t.restart();
  for (size_t i = 0; i < n; ++i) {
    double_table h({p, r});
    table_aggregate<double, double, std::plus<double> >(h, g, h_map, op)();
  }
  cout << "Performed " << n << " aggregates (inl) in " << t.elapsed() << "s." << endl;
}

void time_join_aggregate() {
  boost::timer t;
  cout << "\nJoin-aggregate for fixed sizes:" << endl;

  const size_t p = 10;
  const size_t q = 8;
  const size_t r = 9;

  double_table x({p, q});
  double_table y({q, r});
  std::iota(x.begin(), x.end(), 0);
  std::iota(y.begin(), y.end(), 5);

  finite_index x_map = {0, 1};
  finite_index y_map = {1, 2};
  finite_index r_map = {0, 2};
  finite_index z_shape = {p, q, r};
  std::plus<double> op;

  size_t n = 10000;
  std::multiplies<double> join_op;
  std::plus<double> agg_op;

  t.restart();
  for (size_t i = 0; i < n; ++i) {
    double_table result({p, r});
    table_join_aggregate<double, std::multiplies<double>, std::plus<double> >
      (result, x, y, r_map, x_map, y_map, z_shape, join_op, agg_op).loop();
  }
  cout << "Performed " << n << " join-aggregates (new) in " << t.elapsed() << "s." << endl;

  t.restart();
  for (size_t i = 0; i < n; ++i) {
    double_table result({p, r});
    table_join_aggregate<double, std::multiplies<double>, std::plus<double> >
      (result, x, y, r_map, x_map, y_map, z_shape, join_op, agg_op)();
  }
  cout << "Performed " << n << " join-aggregates (inl) in " << t.elapsed() << "s." << endl;
}

int main(int argc, char** argv) {

  time_join_variable();
  time_join();
  time_aggregate();
  time_join_aggregate();

  return 0;
}

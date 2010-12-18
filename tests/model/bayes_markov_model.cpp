#include <iostream>

#include <boost/array.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/datastructure/dense_table.hpp>
#include <sill/model/bayesian_network.hpp>
// #include <sill/model/free_functions.hpp>
#include <sill/model/markov_network.hpp>

/**
 * \file bayes_markov_model.cpp Bayes net and Markov net test
 */
int main() {

  using namespace sill;
  using namespace std;
  using boost::array;

  // Create a universe.
  universe u;

  /* Create some variables and factors for the Bayes net with this structure:
   * 0, 1 (no parents)
   * 1 --> 2
   * 1,2 --> 3
   * 0,3 --> 4
   */
  finite_variable* x0 = u.new_finite_variable(2);
  finite_variable* x1 = u.new_finite_variable(2);
  finite_variable* x2 = u.new_finite_variable(2);
  finite_variable* x3 = u.new_finite_variable(2);
  finite_variable* x4 = u.new_finite_variable(2);

  finite_var_vector a0 = make_vector(x0);
  array<double, 2> v0 = {{.3, .7}};

  finite_var_vector a1 = make_vector(x1);
  array<double, 2> v1 = {{.5, .5}};

  finite_var_vector a12 = make_vector(x1, x2);
  array<double, 4> v12 = {{.8, .2, .2, .8}};

  finite_var_vector a123 = make_vector(x1, x2, x3);
  array<double, 8> v123 = {{.1, .1, .3, .5, .9, .9, .7, .5}};

  finite_var_vector a034 = make_vector(x0, x3, x4);
  array<double, 8> v034 = {{.6, .1, .2, .1, .4, .9, .8, .9}};

  table_factor f0 = make_dense_table_factor(a0, v0);
  table_factor f1 = make_dense_table_factor(a1, v1);
  table_factor f12 = make_dense_table_factor(a12, v12);
  table_factor f123 = make_dense_table_factor(a123, v123);
  table_factor f034 = make_dense_table_factor(a034, v034);

  bayesian_network<table_factor> bn(make_domain(x0,x1,x2,x3,x4));
  bn.add_factor(x0, f0);
  bn.add_factor(x1, f1);
  bn.add_factor(x2, f12);
  bn.add_factor(x3, f123);
  bn.add_factor(x4, f034);

  cout << "Graph of Bayes net with 5 nodes:\n" << bn << endl;

  markov_network<table_factor> mn = bayes2markov_network(bn);

  cout << "Converted Bayes net to Markov network; graph of Markov net:\n";
  mn.print(cout, true);
  cout << endl;

  finite_assignment a;
  a[x0] = 0;
  a[x2] = 1;
  mn.condition(a);
  cout << "Conditioned Markov net on assignment: " << a << " to get:"
            << endl;
  mn.print(cout, true);
  cout << endl;

  /*
  pairwise_markov_network<table_factor> pmn;
  std::map<finite_variable*, std::vector<finite_variable*> > var_mapping;
  boost::tie(pmn, var_mapping) = fm2pairwise_markov_network(bn, u);

  cout << "Converted Bayes net to pairwise Markov network:\n";
  cout << pmn << endl;
  */
}

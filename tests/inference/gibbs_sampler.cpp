
#include <boost/array.hpp>

#include <sill/base/universe.hpp>
#include <sill/inference/gibbs_sampler.hpp>
#include <sill/model/bayesian_network.hpp>

int main(int argc, char* argv[]) {

  using namespace sill;
  using namespace std;

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
  boost::array<double, 2> v0 = {{.3, .7}};

  finite_var_vector a1 = make_vector(x1);
  boost::array<double, 2> v1 = {{.5, .5}};

  finite_var_vector a12 = make_vector(x1, x2);
  boost::array<double, 4> v12 = {{.8, .2, .2, .8}};

  finite_var_vector a123 = make_vector(x1, x2, x3);
  boost::array<double, 8> v123 = {{.1, .1, .3, .5, .9, .9, .7, .5}};

  finite_var_vector a034 = make_vector(x0, x3, x4);
  boost::array<double, 8> v034 = {{.6, .1, .2, .1, .4, .9, .8, .9}};

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

  size_t nsamples = 100;
  cout << "\nNow taking " << nsamples << " samples from the Bayes net:\n"
       << endl;
  sequential_gibbs_sampler sampler(bn);
  for (size_t i = 0; i < nsamples; ++i)
    cout << sampler.next_sample() << endl;

  return EXIT_SUCCESS;
}

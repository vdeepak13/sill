
//#include <boost/array.hpp>

#include <sill/base/universe.hpp>
#include <sill/graph/grid_graphs.hpp>
#include <sill/inference/gibbs_sampler.hpp>
//#include <sill/model/bayesian_network.hpp>
#include <sill/model/decomposable.hpp>
#include <sill/model/random.hpp>

#include <sill/macros_def.hpp>

using namespace sill;
using namespace std;

void print_error(size_t nsamples,
                 const std::vector<table_factor>& var_marginals,
                 const std::vector<table_factor>& approx_var_marginals);

int main(int argc, char* argv[]) {

  size_t m = 4; // grid height and width
  size_t nsamples = 1000000;
  unsigned random_seed = time(NULL);

  universe u;
  boost::mt19937 rng(random_seed);

  finite_var_vector variables = u.new_finite_variables(m*m, 2);
  pairwise_markov_network<table_factor> mn;
  make_grid_graph(m, m, mn, variables);
  random_ising_model(0.5, 1, mn, rng);

  cout << "Generated random " << m << " x " << m << "pairwise Markov network:\n"
       << mn << endl;

  cout << "\nComputing exact marginals..." << flush;
  std::vector<table_factor> var_marginals;
  {
    decomposable<table_factor> joint;
    joint *= mn.factors();
    foreach(finite_variable* v, variables) {
      var_marginals.push_back(joint.marginal(make_domain(v)));
    }
  }
  cout << "done." << endl;

  cout << "\nNow taking " << nsamples << " samples from the model:\n"
       << endl;
  sequential_gibbs_sampler<table_factor> sampler(mn);
  std::vector<table_factor> approx_var_marginals;
  foreach(finite_variable* v, variables) {
    approx_var_marginals.push_back(table_factor(make_domain(v), 0));
  }
  cout << "Samples\tAvg L1 Error" << endl;
  for (size_t i = 0; i < nsamples; ++i) {
    const finite_assignment& sample = sampler.next_sample();
    for (size_t j = 0; j < variables.size(); ++j) {
      approx_var_marginals[j](sample)++;
    }
    if ((i+1) % (nsamples / 10) == 0) {
      print_error(i+1, var_marginals, approx_var_marginals);
    }
  }

  /*
  cout << "Comparing exact and estimated marginals for each variable\n"
       << "-----------------------------------------------------------\n";
  double avg_L1 = 0; // averaged over variables
  for (size_t j = 0; j < variables.size(); ++j) {
    approx_var_marginals[j].normalize();
    cout << "Var " << j << " truth:\n" << var_marginals[j] << "\n"
         << "Var " << j << " approx:\n" << approx_var_marginals[j] << "\n";
    avg_L1 += norm_1(var_marginals[j], approx_var_marginals[j]);
  }
  avg_L1 /= variables.size();
  cout << "L1 distance between exact and estimated marginals (avg over vars): "
       << avg_L1 << endl;
  */

  return EXIT_SUCCESS;

  /* Create some variables and factors for the Bayes net with this structure:
   * 0, 1 (no parents)
   * 1 --> 2
   * 1,2 --> 3
   * 0,3 --> 4
   */
  /*
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
  */
} // main

void print_error(size_t nsamples,
                 const std::vector<table_factor>& var_marginals,
                 const std::vector<table_factor>& approx_var_marginals_) {
  std::vector<table_factor> approx_var_marginals(approx_var_marginals_);
  double avg_L1 = 0; // averaged over variables
  for (size_t j = 0; j < var_marginals.size(); ++j) {
    approx_var_marginals[j].normalize();
    avg_L1 += norm_1(var_marginals[j], approx_var_marginals[j]);
  }
  avg_L1 /= var_marginals.size();
  cout << nsamples << "\t" << avg_L1 << endl;
}

#include <sill/macros_def.hpp>

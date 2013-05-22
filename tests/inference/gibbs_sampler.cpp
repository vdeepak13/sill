
#include <boost/timer.hpp>

#include <sill/base/universe.hpp>
#include <sill/graph/grid_graphs.hpp>
#include <sill/inference/gibbs_sampler.hpp>
#include <sill/model/decomposable.hpp>
#include <sill/model/random.hpp>

#include <sill/macros_def.hpp>

using namespace sill;
using namespace std;

double get_error(const std::vector<table_factor>& var_marginals,
                 const std::vector<table_factor>& approx_var_marginals);

int main(int argc, char* argv[]) {

  size_t m = 4; // grid height and width
  size_t nsamples = 10000000;
  unsigned random_seed = 2390249;

  universe u;
  boost::mt19937 rng(random_seed);

  finite_var_vector variables = u.new_finite_variables(m*m, 2);
  pairwise_markov_network<table_factor> mn;
  make_grid_graph(m, m, mn, variables);
  random_ising_model(0.5, 1, mn, rng);

  cout << "Generated random " << m << " x " << m << "pairwise Markov network."
       << endl;

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

  cout << "\nNow taking " << nsamples << " samples from the model..."
       << flush;
  sequential_gibbs_sampler<table_factor> sampler(mn);
  std::vector<table_factor> approx_var_marginals;
  foreach(finite_variable* v, variables) {
    approx_var_marginals.push_back(table_factor(make_domain(v), 0));
  }
  std::vector<std::pair<size_t, double> > avg_L1_errors;
  boost::timer t;
  for (size_t i = 0; i < nsamples; ++i) {
    const finite_record& sample = sampler.next_sample();
    for (size_t j = 0; j < variables.size(); ++j) {
      approx_var_marginals[j](sample)++;
    }
    if ((i+1) % (nsamples / 10) == 0) {
      avg_L1_errors.push_back
        (std::make_pair(i, get_error(var_marginals, approx_var_marginals)));
    }
  }
  cout << "done in " << t.elapsed() << " seconds." << endl;

  cout << "Samples\tAvg L1 Error" << endl;
  for (size_t i = 0; i < avg_L1_errors.size(); ++i) {
    cout << avg_L1_errors[i].first << "\t" << avg_L1_errors[i].second << "\n";
  }
  cout << endl;

  return EXIT_SUCCESS;

} // main

double get_error(const std::vector<table_factor>& var_marginals,
                 const std::vector<table_factor>& approx_var_marginals_) {
  std::vector<table_factor> approx_var_marginals(approx_var_marginals_);
  double avg_L1 = 0; // averaged over variables
  for (size_t j = 0; j < var_marginals.size(); ++j) {
    approx_var_marginals[j].normalize();
    avg_L1 += norm_1(var_marginals[j], approx_var_marginals[j]);
  }
  avg_L1 /= var_marginals.size();
  return avg_L1;
}

#include <sill/macros_def.hpp>

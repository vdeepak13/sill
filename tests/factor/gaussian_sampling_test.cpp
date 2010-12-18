#include <iostream>

#include <sill/base/universe.hpp>
#include <sill/model/random.hpp>

#include <sill/macros_def.hpp>

/**
 * \file gaussian_sampling_test.cpp  Test sampling.
 *
 * Generate a multivariate Gaussian and compare:
 *  - the true entropy of the distribution
 *  - the cross entropy between the sample distribution and the truth
 * Generate a decomposable Gaussian model and compare:
 *  - the true entropy of the model
 *  - the cross entropy between the sample distribution and the truth
 */

int main(int argc, char** argv) {

  using namespace sill;
  using namespace std;

  // Dataset parameters
  size_t nsamples = 10000;
  size_t n = 25; // length of width-2 chain decomposable model
  unsigned model_seed = 3078574;
  unsigned oracle_seed = 1284392;

  universe u;
  boost::mt11213b rng(oracle_seed);

  // Create a multivariate Gaussian.
  boost::uniform_real<double> unif_real(-5, 5);
  vector_var_vector X;
  for (size_t j(0); j < 5; ++j)
    X.push_back(u.new_vector_variable(1));
  size_t Xsize(vector_size(X));
  vec mu(Xsize, 0);
  foreach(double& val, mu)
    val = unif_real(rng);
  mat sigma(Xsize, Xsize, 1.);
  sigma += identity(Xsize);
  canonical_gaussian cg(moment_gaussian(X, mu, sigma));

  // Sample
  double true_entropy(cg.entropy());
  double cross_entropy(0);
  cout << "Made multivariate Gaussian.\n"
       << "Sampling and comparing true entropy with cross entropy:\n"
       << "samples\tTruth\tX entropy\tDifference\n"
       << "---------------------------------------------------" << endl;
  for (size_t i(0); i < nsamples; ++i) {
    vector_assignment a(cg.sample(rng));
    cross_entropy -= cg.logv(a);
    if (i % (nsamples / 10) == 0) {
      double estimate(cross_entropy / (i+1.));
      cout << (i+1) << "\t" << true_entropy << "\t" << estimate << "\t"
           << (true_entropy - estimate) << endl;
    }
  }

  // Create a model to sample from
  decomposable<canonical_gaussian> YXmodel;
  crf_model<gaussian_crf_factor> YgivenXmodel;
  boost::tuple<vector_var_vector, vector_var_vector,
               std::map<vector_variable*, copy_ptr<vector_domain> > >
    Y_X_and_map(create_chain_gaussian_crf(YXmodel, YgivenXmodel, n, u,
                                          model_seed));

  // Sample
  true_entropy = YXmodel.entropy();
  cross_entropy = 0;
  cout << "Made Gaussian decomposable model.\n"
       << "Sampling and comparing true entropy with cross entropy:\n"
       << "samples\tTruth\tX entropy\tDifference\n"
       << "---------------------------------------------------" << endl;
  for (size_t i(0); i < nsamples; ++i) {
    vector_assignment a(YXmodel.sample(rng));
    cross_entropy -= YXmodel.log_likelihood(a);
    if (i % (nsamples / 10) == 0) {
      double estimate(cross_entropy / (i+1.));
      cout << (i+1) << "\t" << true_entropy << "\t" << estimate << "\t"
           << (true_entropy - estimate) << endl;
    }
  }

  return 0;
}

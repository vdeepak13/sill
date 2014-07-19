#include <boost/timer.hpp>

#include <sill/learning/dataset2/finite_dataset.hpp>
#include <sill/learning/parameter/table_factor_learner.hpp>
#include <sill/learning/structure/chow_liu.hpp>
#include <sill/model/random.hpp>

#include <sill/macros_def.hpp>

/**
 * Time our Chow-Liu implementation on a random HMM.
 * 
 * release timings:
 * Chow-Liu using the assignment_dataset (old): 0.551218 s/trial
 * Chow-Liu using the vector_dataset (old): 0.0766336 s/trial
 * Chow-Liu using the finite_dataset (new): 0.0260711 s/trial
 */
int main(int argc, char* argv[]) {

  using namespace sill;
  using namespace std;

  // timing parameters
  size_t nsamples = argc > 1 ? atol(argv[1]) : 1000;
  size_t ntrain   = argc > 2 ? atol(argv[2]) : 1;
  size_t ntest    = argc > 3 ? atol(argv[3]) : 1;

  // generate a random model
  size_t n = 10;
  size_t n_states = 2;
  size_t n_emissions = 2;
  unsigned random_seed = 69371053;
  unsigned o_random_seed = 120592;

  universe u;
  boost::mt11213b rng(random_seed);
  bayesian_network<table_factor> bn;
  random_HMM(bn, rng, u, n, n_states, n_emissions);
  cout << bn << endl;

  // generate some data from this model
  finite_dataset<> ds;
  ds.initialize(bn.arguments());
  for (size_t i = 0; i < nsamples; ++i) {
    ds.insert(bn.sample(rng));
  }

  // time the learning
  boost::timer ttrain;
  decomposable<table_factor> dm;
  for (size_t i = 0; i < ntrain; ++i) {
    cout << "Trial " << i << endl;
    table_factor_learner<> learner(ds);
    chow_liu<table_factor> chowliu(ds.arguments(), learner);
    dm = chowliu.model();
  }
  cout << "Chow-Liu using the new dataset: "
       << ttrain.elapsed() / ntrain << " s/trial" << endl;

//   // time the log-likelihood evaluation
//   boost::timer ttest;
//   double ll = 0;
//   for (size_t i = 0; i < ntest; ++i) {
//     cout << "Trial " << i << endl;
//     foreach(const record<>& rec, ds.records()) {
//       ll += dm.log_likelihood(rec);
//     }
//   }
//   cout << "Log-likelihood evaluation: "
//        << ttest.elapsed() / ntest << " s/trial" << endl;

}

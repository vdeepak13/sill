#define BOOST_TEST_MODULE mixture_em
#include <boost/test/unit_test.hpp>

#include <boost/math/special_functions/round.hpp>
#include <boost/random/lagged_fibonacci.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/tuple/tuple_io.hpp>
#include <boost/tuple/tuple_comparison.hpp>

#include <sill/base/universe.hpp>

#include <sill/learning/dataset/vector_memory_dataset.hpp>
#include <sill/learning/factor_mle/moment_gaussian.hpp>
#include <sill/learning/parameter/mixture_em.hpp>

#include <algorithm>

boost::mt19937 rng; 

template class sill::mixture_em<sill::moment_gaussian>;

BOOST_AUTO_TEST_CASE(test_convergence) {
  using namespace sill;
  using namespace std;
  
  size_t k = 3;
  size_t nsamples = 2000;
  size_t niters = 50;
  // size_t nsamples = 10000;
  // size_t niters = 100;

  universe u;
  vector_var_vector var_vec = u.new_vector_variables(1, 2); // 1 2D variable
  vector_domain vars = make_domain(var_vec);
  
  mixture_gaussian original(k, vars);
  original[0] = moment_gaussian(var_vec, "-2 0", "1 0.5; 0.5 1");
  original[1] = moment_gaussian(var_vec, "2 -2", "1 0.2; 0.2 1");
  original[2] = moment_gaussian(var_vec, "2 +2", "1 -0.2; -0.2 1");

  vector_memory_dataset<> data;
  data.initialize(var_vec, nsamples);

  boost::lagged_fibonacci607 rng;
  for (size_t i = 0; i < nsamples; ++i) {
    data.insert(original.sample(rng));
  }


  mixture_em<moment_gaussian> engine(k, vars);
//   engine.initialize(&data);
//   for(size_t i = 0; i < niters; ++i) {
//     double log_lik = engine.iterate();
//     cout << "Iteration " << i << ", log-likelihood " << log_lik << endl;
//   }
  mixture_em<moment_gaussian>::param_type params;
  params.verbose = true;
  params.seed = 123;

  mixture_gaussian estimate;
  engine.learn(data, params, estimate);

  // retrieve the components in the canonical order
  std::vector<boost::tuple<double,double,size_t> > centers(k);
  for(size_t i = 0; i < k; ++i) {
    using boost::math::round;
    vec mean = estimate[i].mean();
    centers[i] = boost::make_tuple(round(mean[0]), round(mean[1]), i);
    // cout << mean.t() << ": " <<  centers[i] << endl;
  }
  std::sort(centers.begin(), centers.end());

  for (size_t i = 0; i < k; ++i) {
    size_t j = boost::get<2>(centers[i]);
    double kl = original[i].relative_entropy(estimate[j]);
    cout << i << " " << j << ": "
         << estimate[j].mean().t() << "\t" << kl << std::endl;
    BOOST_CHECK_SMALL(kl, 0.02);
  }
}

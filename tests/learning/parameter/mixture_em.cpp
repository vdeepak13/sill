//#define DS2

#define BOOST_TEST_MODULE mixture_em
#include <boost/test/unit_test.hpp>

#include <boost/math/special_functions/round.hpp>
#include <boost/random/lagged_fibonacci.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/tuple/tuple_io.hpp>
#include <boost/tuple/tuple_comparison.hpp>

#include <sill/base/universe.hpp>

#ifdef DS2
  #include <sill/factor/random/moment_gaussian_data.hpp>
  #include <sill/learning/dataset2/vector_dataset.hpp>
  #include <sill/learning/parameter/mixture_em.hpp>
  #include <sill/learning/parameter/moment_gaussian_mle.hpp>
#else
  #include <sill/learning/dataset3/vector_memory_dataset.hpp>
  #include <sill/learning/mle/moment_gaussian.hpp>
  #include <sill/learning/parameter/mixture_em3.hpp>
#endif

#include <algorithm>

boost::mt19937 rng; 

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

#ifdef DS2  
  vector_dataset<> data;
  data.initialize(var_vec);
#else
  vector_memory_dataset<> data;
  data.initialize(var_vec, nsamples);
#endif

  boost::lagged_fibonacci607 rng;
  for (size_t i = 0; i < nsamples; ++i) {
    data.insert(original.sample(rng));
  }

#ifdef DS2
  typedef mixture_em<moment_gaussian, vector_dataset<> > em_engine;

  moment_gaussian_mle<> learner(&data);
  moment_gaussian_data<> generator(&data);
  em_engine engine(&data, learner, k, vars, generator);
  cout << engine.estimate() << endl;
#else
  mixture_gaussian init;
  initialize_em(data, k, init);
  mixture_em<moment_gaussian> engine(&data, init);
#endif
  
  for(size_t i = 1; i <= niters; i++) {
    double log_lik = engine.expectation();
    engine.maximization();
    cout << "Iteration " << i << ", log-likelihood " << log_lik << endl;
    //cout << "\t" << engine.estimate() << endl;
  }

  // retrieve the components in the canonical order
  mixture_gaussian estimate = engine.estimate();
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

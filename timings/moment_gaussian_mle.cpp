#define DS2

#include <boost/timer.hpp>

#include <sill/base/universe.hpp>

#ifdef DS2
  #include <sill/learning/dataset2/vector_dataset.hpp>
  #include <sill/learning/parameter/moment_gaussian_mle.hpp>
#else
  #include <sill/learning/dataset3/vector_memory_dataset.hpp>
  #include <sill/learning/mle/moment_gaussian.hpp>
#endif

#include <sill/macros_def.hpp>

int main(int argc, char** argv) {

  using namespace sill;
  using namespace std;
  
  // parameters
  size_t nsamples = argc > 1 ? atol(argv[1]) : 1000;
  size_t ntrain = argc > 2 ? atol(argv[2]) : 1;

  // create a distribution
  universe u;
  vector_var_vector v = u.new_vector_variables(3, 1);
  //  moment_gaussian mg(v, "0.5 1", "2 1; 1 2");
   moment_gaussian mg(v, "0.5 1 2", "3 2 1; 2 2 1; 1 1 2");

  // generate some data from this distribution
  boost::lagged_fibonacci607 rng;
#ifdef DS2
  vector_dataset<> ds;
  ds.initialize(v);
#else
  vector_memory_dataset<> ds;
  ds.initialize(v, nsamples);
#endif

  for (size_t i = 0; i < nsamples; ++i) {
    ds.insert(mg.sample(rng));
  }
  cout << "Inserted " << ds.size() << " samples" << endl;

  // time the learning
  boost::timer timer;
  for (size_t i = 0; i < ntrain; ++i) {
#ifdef DS2
    moment_gaussian_mle<> estim(&ds);
#else
    mle<moment_gaussian> estim(&ds);
#endif
    estim(make_domain(v));
  }
  cout << "Time per estimation: " << timer.elapsed() / ntrain << " s/trial" << endl;
}

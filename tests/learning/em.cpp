#include <boost/random/mersenne_twister.hpp>
#include <boost/lexical_cast.hpp>

#include <sill/base/universe.hpp>
#include <sill/learning/em_mog.hpp>
#include <sill/learning/dataset/assignment_dataset.hpp>
#include <sill/learning/dataset/data_loader.hpp>
#include <sill/range/numeric.hpp>

boost::mt19937 rng; 

int main(int argc, char** argv) {
  using namespace sill;
  using namespace std;
  
  typedef em_mog em_engine;

  string filename("../../../tests/data/mixture.txt");
  size_t k = 2;
  size_t niters = (argc > 1) ? boost::lexical_cast<size_t>(argv[1]) : 10;
  double regul = 1e-8;
  
  universe u;
  vector_var_vector v = u.new_vector_variables(1, 2); // 1 2D variable

  assignment_dataset<> data =
    *(data_loader::load_plain<assignment_dataset<> >
      (filename, finite_var_vector(), v,
       std::vector<variable::variable_typenames>()));
  cout << data << endl;

  em_engine engine(&data, k);
  mixture_gaussian estimate = engine.initialize(rng, regul);
  cout << estimate << endl;

  for(size_t i = 1; i <= niters; i++) {
    double log_lik = engine.expectation(estimate);
    estimate = engine.maximization(regul);
    cout << "Iteration " << i << ", log-likelihood " << log_lik << endl;
    cout << "\t" << estimate << endl;
  }
}

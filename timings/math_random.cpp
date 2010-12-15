#include <fstream>
#include <sstream>

#include <boost/random/mersenne_twister.hpp>
#include <boost/timer.hpp>

#include <prl/math/random.hpp>

#include <prl/macros_def.hpp>

using namespace prl;

/**
 * Pass in parameter '1' in order to print out a bunch of numbers from various
 * distributions to see if this is working properly.
 */
int main(int argc, char* argv[]) {

  bool print_stuff = false;
  if (argc == 2 && atoi(argv[1]) == 1)
    print_stuff = true;

  boost::timer timer;

  std::vector<double> gamma_k;
  gamma_k.push_back(.01);
  gamma_k.push_back(.1);
  gamma_k.push_back(1);
  gamma_k.push_back(10);
  std::vector<double> gamma_alpha;
  gamma_alpha.push_back(.1);
  gamma_alpha.push_back(1);
  gamma_alpha.push_back(10);

  size_t nsamples = 1000000;
  std::vector<double> samples(nsamples, 0);
  unsigned random_seed = 69371053;
  boost::mt11213b rng(random_seed);

  std::cout << "Generating " << nsamples << " samples from Gamma(k, alpha):\n"
            << "k\talpha\tavg time to sample" << std::endl;

  foreach(double k, gamma_k) {
    foreach(double alpha, gamma_alpha) {
      gamma_distribution<> gamma(k, alpha);
      timer.restart();
      for (size_t i = 0; i < nsamples; ++i)
        samples[i] = gamma(rng);
      std::cout << k << "\t" << alpha << "\t" << (timer.elapsed() / nsamples)
                << std::endl;
      if (print_stuff) {
        std::ostringstream os;
        os << "gamma_samples_" << k << "_" << alpha << ".txt";
        std::ofstream out(os.str().c_str(), std::ios::out);
        foreach(double val, samples) {
          out << val << "\n";
        }
        out.close();
      }
    }
  }

  size_t dirichlet_n = 10;
  nsamples = 10000;

  std::cout << "Generating " << nsamples
            << " samples from Dirichlet(alpha) with n = " << dirichlet_n << ":\n"
            << "alpha\tavg time to sample" << std::endl;

  foreach(double k, gamma_k) {
    dirichlet_distribution<> dirichlet(dirichlet_n, k);
    timer.restart();
    vec sample;
    for (size_t i = 0; i < nsamples; ++i) {
      sample = dirichlet(rng);
      if (argc == 1000)
        std::cout << sample;
    }
    std::cout << k << "\t" << (timer.elapsed() / nsamples) << std::endl;
  }

}

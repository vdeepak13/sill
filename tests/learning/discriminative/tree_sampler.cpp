#include <iostream>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/timer.hpp>

#include <prl/learning/discriminative/tree_sampler.hpp>

#include <prl/macros_def.hpp>

int main(int argc, char* argv[]) {

  using namespace prl;
  using namespace std;

  size_t nvals = 11;
  size_t nsamples = 1000000;
  boost::timer t;

  std::vector<double> distrib(nvals,0);
  boost::mt11213b rng(2937562);
  boost::uniform_real<double> uniform_prob(0,1);
  foreach(double& d, distrib)
    d = uniform_prob(rng);

  tree_sampler r(distrib, nvals);
  r.write(cout);
  cout << endl;

  cout << "Sampling " << nsamples << " samples...";
  std::vector<double> hist(nvals, 0);
  t.restart();
  for (size_t i = 0; i < nsamples; ++i)
    ++(hist[r.sample()]);
  cout << "done in " << t.elapsed() / nsamples << " seconds" << endl;

  double total = 0;
  foreach(double d, hist)
    total += d;
  foreach(double& d, hist)
    d /= total;

  cout << "Distribution over samples: " << hist << endl;

  tree_sampler r2(nvals);
  for (size_t i = 0; i < nvals; ++i)
    r2.set(i, distrib[i]);
  r2.commit_update();
  cout << "\nSampler made using other constructor: " << r2 << endl;
}


#include <set>
#include <sstream>
#include <vector>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/timer.hpp>

#include <sill/boost_unordered_utils.hpp>
#include <sill/math/statistics.hpp>

#include <sill/macros_def.hpp>

using namespace sill;

//! For small n
template <typename SetType>
double simple_time_set(const std::vector<size_t>& vals, size_t n, size_t runs);

//! For large n
template <typename SetType>
std::pair<double, double>
fancy_time_set(const std::vector<size_t>& vals, size_t n, size_t runs);

int main(int argc, char** argv) {

  size_t n = 10000000;
  if (argc == 2) {
    std::istringstream is(argv[1]);
    assert(is >> n);
  }

  unsigned random_seed = time(NULL);

  boost::mt11213b rng(random_seed);
  boost::uniform_int<int> unif_int(0,std::numeric_limits<int>::max());
  std::vector<size_t> vals;
  vals.reserve(n);
  for (size_t i = 0; i < n; ++i)
    vals.push_back(unif_int(rng));

  std::vector<size_t> runs;
  std::vector<size_t> ns;
  runs.push_back(1000000); ns.push_back(1);
  runs.push_back(500000); ns.push_back(2);
  runs.push_back(250000); ns.push_back(4);
  runs.push_back(100000); ns.push_back(8);
  runs.push_back(50000); ns.push_back(16);
  runs.push_back(10000); ns.push_back(32);
  runs.push_back(1000); ns.push_back(100);

  std::cout << "Timing sets (millisec):\n"
            << "-----------------------------------------\n"
            << "n\tstd::set\tboost::unordered_set\tstderr\tstderr\n"
            << "------------------------------------------------------------"
            << std::endl;
  for (size_t k = 0; k < runs.size(); ++k) {
    double std_time =
      simple_time_set<std::set<size_t> >(vals, ns[k], runs[k]);
    double boost_time =
      simple_time_set<boost::unordered_set<size_t> >(vals, ns[k], runs[k]);
    std::cout << ns[k] << "\t" << std_time << "\t" << boost_time << "\n";
  }
  {
    std::pair<double,double> std_time =
      fancy_time_set<std::set<size_t> >(vals, n/10, 10);
    std::pair<double,double> boost_time =
      fancy_time_set<boost::unordered_set<size_t> >(vals, n/10, 10);
    std::cout << (n/10) << "\t" << std_time.first << "\t" << boost_time.first
              << "\t" << std_time.second << "\t" << boost_time.second << "\n";
    std_time = fancy_time_set<std::set<size_t> >(vals, n, 1);
    boost_time = fancy_time_set<boost::unordered_set<size_t> >(vals, n, 1);
    std::cout << n << "\t" << std_time.first << "\t" << boost_time.first
              << "\t" << std_time.second << "\t" << boost_time.second << "\n";
  }

} // main

template <typename SetType>
double simple_time_set(const std::vector<size_t>& vals, size_t n, size_t runs) {
  boost::timer timer;
  for (size_t run = 0; run < runs; ++run) {
    SetType s;
    for (size_t i = 0; i < n; ++i) {
      s.insert(vals[i]);
    }
    if (vals[0] == 0)
      std::cout << (*(s.begin())) << std::endl;
  }
  double elapsed = timer.elapsed();
  return (elapsed * (1000./ (double)runs));
} // simple_time_set

template <typename SetType>
std::pair<double, double>
fancy_time_set(const std::vector<size_t>& vals, size_t n, size_t runs) {
  std::vector<double> times(runs, 0);
  boost::timer timer;
  for (size_t run = 0; run < runs; ++run) {
    timer.restart();
    {
      SetType s;
      for (size_t i = 0; i < n; ++i) {
        s.insert(vals[i]);
      }
      if (vals[0] == 0)
        std::cout << (*(s.begin())) << std::endl;
    }
    times[run] = 1000. * timer.elapsed();
  }
  return mean_stderr(times);
} // fancy_time_set

#include <sill/macros_undef.hpp>

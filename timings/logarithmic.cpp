#include <iostream>
#include <string>

#include <sill/math/logarithmic.hpp>
#include <cmath>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/progress.hpp>

int main(int argc, char** argv)
{
  using namespace std;
  using namespace boost;

  typedef sill::logarithmic<double> log_double;

  // Compare the time required for addition/multiplication in the
  // regular and log spaces.
  //
  // Using GCC 4.0 I got:
  //
  // Performed 100000000 random samplings in 0.65 seconds.
  // Performed 100000000 log-space conversions in 13.48 seconds.
  // Performed 100000000 additions in 0.01 seconds.
  // Performed 100000000 additions in log space in 23.81 seconds.
  // Performed 100000000 multiplications in -0.06 seconds.
  // Performed 100000000 multiplications in log space in -0.12 seconds.
  //
  // So multiplication is a little faster in log space (since it's
  // implemented using addition), but addition is much, much slower.
  const int N = 100000000;
  double sampling_time = 0.0;
  {
    timer t;
    mt19937 rng;
    uniform_01<mt19937, double> unif01(rng);
    double y;
    for (int i = 0; i < N; ++i) {
      y = unif01();
    }
    if (argc > 10000)
      cout << y;
    sampling_time = t.elapsed();
    cout << "Performed " << N << " random samplings in "
         << sampling_time << " seconds." << endl;
  }

  double conversion_time = 0.0;
  {
    timer t;
    mt19937 rng;
    uniform_01<mt19937, double> unif01(rng);
    for (int i = 0; i < N; ++i) {
      log_double y;
      y = unif01();
    }
    conversion_time = t.elapsed() - sampling_time;
    cout << "Performed " << N << " log-space conversions in "
              << conversion_time << " seconds." << endl;
  }

  {
    double x = 0.0;
    timer t;
    mt19937 rng;
    uniform_01<mt19937, double> unif01(rng);
    for (int i = 0; i < N; ++i) {
      double y = unif01();
      x += y;
    }
    double time = t.elapsed() - sampling_time;
    cout << "Performed " << N << " additions in "
              << time << " seconds." << endl;
  }

  {
    log_double x;
    x = 0.0;
    timer t;
    mt19937 rng;
    uniform_01<mt19937, double> unif01(rng);
    for (int i = 0; i < N; ++i) {
      log_double y;
      y = unif01();
      x += y;
    }
    double time = t.elapsed() - conversion_time - sampling_time;
    cout << "Performed " << N << " additions in log space in "
              << time << " seconds." << endl;
  }

  {
    double x = 1.0;
    timer t;
    mt19937 rng;
    uniform_01<mt19937, double> unif01(rng);
    for (int i = 0; i < N; ++i) {
      double y = 2.0 * unif01();
      x *= y;
    }
    double time = t.elapsed() - sampling_time;
    cout << "Performed " << N << " multiplications in "
              << time << " seconds." << endl;
  }

  {
    log_double x;
    x = 1.0;
    timer t;
    mt19937 rng;
    uniform_01<mt19937, double> unif01(rng);
    for (int i = 0; i < N; ++i) {
      log_double y;
      y = 2.0 * unif01();
      x *= y;
    }
    double time = t.elapsed() - conversion_time - sampling_time;
    cout << "Performed " << N << " multiplications in log space in "
              << time << " seconds." << endl;
  }

  // Report success.
  return EXIT_SUCCESS;
}

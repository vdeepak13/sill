#include <sill/factor/mixture.hpp>
#include <sill/factor/moment_gaussian.hpp>
#include <sill/math/linear_algebra.hpp>

namespace sill {

  moment_gaussian project(const mixture_gaussian& mixture) {
    // Do moment matching
    vector_var_vector args = make_vector(mixture.arguments());
    size_t n = mixture[0].size();
    double norm = mixture.norm_constant();
    assert(norm > 0);

    // Match the mean
    vec mean = zeros(n);
    // std::cout << mean << std::endl;
    for(size_t i = 0; i < mixture.size(); i++) {
      double w = mixture[i].norm_constant() / norm;
      mean += w * mixture[i].mean(args);
      // std::cout << mean << std::endl;
    }

    // Match the covariance
    mat cov = zeros(n, n);
    for(size_t i = 0; i < mixture.size(); i++) {
      double w = mixture[i].norm_constant() / norm;
      vec x = mixture[i].mean(args) - mean;
      cov += w * (mixture[i].covariance(args) + outer_product(x, x));
    }
    return moment_gaussian(args, mean, cov, norm);
  }

} // namespace sill


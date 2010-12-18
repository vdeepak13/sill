#include <cmath>

#include <sill/learning/parameter/gaussian.hpp>
#include <sill/learning/dataset/dataset.hpp>
#include <sill/math/linear_algebra.hpp>

#include <sill/macros_def.hpp>

namespace sill {
  
  void mle(const dataset& data, const vec& w, double regul, moment_gaussian& mg)
  {
    using std::log;

    size_t n = data.vector_dim();
    size_t i = 0; // datapoint index
    double sumw = sum(w);

    // compute the mean
    vec ctr(n, 0);
    foreach(const record& rec, data.records())
      ctr += w[i++] * rec.vector();
    ctr /= sumw;

    // compute the covariance
    i = 0;
    mat cov(n, n);
    cov.clear();
    foreach(const record& rec, data.records()) {
      vec x = rec.vector() - ctr;
      cov += w[i++] * outer_product(x, x);
    }
    cov /= sumw;

    // add regularization
    if (regul > 0) cov += identity(n) * regul;

    // return the resulting factor
    mg = moment_gaussian(data.vector_list(), ctr, cov, sumw);
  }

  void mle(const dataset& data, double reg, moment_gaussian& mg) {
    mle(data, ones(data.size()), reg, mg);
  }

} // namespace sill

#include <boost/math/distributions/normal.hpp>

#include <sill/base/stl_util.hpp>
#include <sill/factor/approx/hybrid_conditional.hpp>
#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/mixture.hpp>
#include <sill/factor/moment_gaussian.hpp>
#include <sill/factor/nonlinear_gaussian.hpp>
#include <sill/factor/operations.hpp>
#include <sill/math/linear_algebra.hpp>

namespace sill {

  hybrid_conditional_approximator::
  hybrid_conditional_approximator(const gaussian_approximator& base,
                                  vector_variable* split_var,
                                  size_t npoints,
                                  double minstdev,
                                  double nstdevs,
                                  double max_range)
    : approx_ptr(base.clone()),
      split_var(split_var),
      npoints(npoints),
      minstdev(minstdev),
      nstdevs(nstdevs),
      max_range(max_range) {
    assert(split_var->size() == 1);
  }

  moment_gaussian hybrid_conditional_approximator::
  operator()(const nonlinear_gaussian& ng, const moment_gaussian& prior) const {
    using std::sqrt;
    assert(prior.arguments().count(split_var));
    double stdev = sqrt(prior.covariance(split_var)(0,0));
    double mean = prior.mean(split_var)(0);
    boost::math::normal normal(mean, stdev);
    if (stdev <= minstdev) {
      //std::cerr << "standard approx" << std::endl;
      return approx()(ng, prior);
    } else {
      std::cerr << "hybrid approx" << std::endl;
      // Approximate the prior with a mixture at regular points of split_var
      size_t n = npoints;
      mixture< moment_gaussian > mix(n, set_union(prior.arguments(), ng.arguments()));
      double norm = 0;
      vec p; // the integration points
      if (stdev*nstdevs < max_range / 2) {
        p = linspace(mean - stdev*nstdevs, mean + stdev*nstdevs, n);
      } else {
        p = linspace(mean - max_range/2, mean + max_range/2, n);
      }
      for(size_t i = 0; i < n; i++) {
        vector_assignment a;
        a[split_var] = vec_1(p[i]);
        // compute the prior restricted to the integration point p[i]
        moment_gaussian restricted = prior.restrict(a);
        if (max_range != inf()) {
          restricted.likelihood += pdf(normal, p[i] + max_range);
          restricted.likelihood += pdf(normal, p[i] - max_range);
        }
        // approximate the joint at the integration point p[i]
        moment_gaussian fi = approx()(ng.restrict(a), restricted);
        assert(fi.arguments().count(split_var) == 0);
        mix[i] = fi * moment_gaussian(make_vector(split_var),
                                      vec(1, p[i]),
                                      mat(1, 1, 1e-8));
        //std::cerr << mix[i] << std::endl;
        norm += restricted.likelihood;
      }
      assert(norm > 0);
      mix *= constant_factor(prior.likelihood / norm);
      //std::cout << mix << std::endl;
      for(size_t i = 0; i < mix.size(); i++)
        std::cerr << p[i] << ':' << mix[i].likelihood << ' ';
      std::cerr << std::endl;
      moment_gaussian result = project(mix);
      // std::cerr << "Projection: " << result << std::endl;
      // TODO: ensure that f \preceq prior
      return result;
    }
  }

} // namespace sill


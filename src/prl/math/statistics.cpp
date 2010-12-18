
#include <sill/math/is_finite.hpp>
#include <sill/math/statistics.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Means, standard errors, medians, and MADs
  //============================================================================

  std::pair<double, double> mean_stderr(const std::vector<double>& vals) {
    double sum(0);
    double sum2(0);
    if (vals.size() == 0)
      return std::make_pair(0, 0);
    foreach(double d, vals) {
      if (!is_finite(d))
        return std::make_pair(d,std::numeric_limits<double>::infinity());
      sum += d;
      sum2 += d * d;
    }
    double mean(sum / vals.size());
    double stderror(std::sqrt(sum2 / vals.size() - mean * mean)
                    / std::sqrt(vals.size()));
    return std::make_pair(mean, stderror);
  }

  std::pair<double, double> mean_stderr(const vec& vals) {
    double sum(0);
    double sum2(0);
    if (vals.size() == 0)
      return std::make_pair(0, 0);
    foreach(double d, vals) {
      if (!is_finite(d))
        return std::make_pair(d,std::numeric_limits<double>::infinity());
      sum += d;
      sum2 += d * d;
    }
    double mean(sum / vals.size());
    double stderror(std::sqrt(sum2 / vals.size() - mean * mean)
                    / std::sqrt(vals.size()));
    return std::make_pair(mean, stderror);
  }

  std::pair<double, double> median_MAD(const std::vector<double>& vals) {
    assert(vals.size() != 0);
    std::vector<double> sorted(vals);
    std::sort(sorted.begin(), sorted.end());
    size_t median_i(sorted.size() / 2);
    double median(sorted[median_i]);
    foreach(double& v, sorted)
      v = fabs(v - median);
    std::sort(sorted.begin(), sorted.end());
    return std::make_pair(median, sorted[median_i]);
  }

  std::pair<double, double> median_MAD(const vec& vals) {
    assert(vals.size() != 0);
    vec sorted(vals);
    std::sort(sorted.begin(), sorted.end());
    size_t median_i(sorted.size() / 2);
    double median(sorted[median_i]);
    foreach(double& v, sorted)
      v = fabs(v - median);
    std::sort(sorted.begin(), sorted.end());
    return std::make_pair(median, sorted[median_i]);
  }

  // Order statistics
  //============================================================================

  std::vector<size_t> sorted_indices(const vec& v) {
    std::vector<size_t> ind(v.size());
    for (size_t i(0); i < v.size(); ++i)
      ind[i] = i;
    impl::sorted_indices_comparator comp(v);
    std::sort(ind.begin(), ind.end(), comp);
    return ind;
  }

} // end of namespace: prl

#include <sill/macros_undef.hpp>

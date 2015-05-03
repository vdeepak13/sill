#ifndef SILL_MATH_STATISTICS_HPP
#define SILL_MATH_STATISTICS_HPP

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>

#include <sill/math/is_finite.hpp>
#include <sill/math/linear_algebra/armadillo.hpp>
#include <sill/math/operations.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Sums, means, standard errors, medians, and MADs
  //============================================================================

  //! Return the sum of the given vector of values.
  template <typename T>
  double sum(const std::vector<T>& v) {
    double sum = 0;
    foreach(T val, v)
      sum += val;
    return sum;
  }

  //! Return the mean for the given vector of values.
  //! Return 0 for empty vector.
  template <typename T>
  double mean(const std::vector<T>& v) {
    if (v.size() == 0)
      return 0;
    return sum(v) / v.size();
  }

  /*
  //! Return the mean for the given vector of values.
  //! Return 0 for empty vector.
  template <typename T>
  double mean(const arma::Col<T>& v) {
    if (v.size() == 0)
      return 0;
    double sum = 0;
    for (size_t i = 0; i < v.size(); ++i)
      sum += v[i];
    return sum / v.size();
  }
  */

  //! Return the median for the given vector of values.
  //! Return 0 for empty vector.
  template <typename T>
  T median(const std::vector<T>& v) {
    if (v.size() == 0)
      return 0;
    std::vector<T> sorted(v);
    std::sort(sorted.begin(), sorted.end());
    size_t median_i(sorted.size() / 2);
    return sorted[median_i];
  }

  /*
  //! Return the median for the given vector of values.
  //! Return 0 for empty vector.
  template <typename T>
  T median(const arma::Col<T>& v) {
    if (v.size() == 0)
      return 0;
    arma::Col<T> sorted(v);
    std::sort(sorted.begin(), sorted.end());
    size_t median_i(sorted.size() / 2);
    return sorted[median_i];
  }
  */

  //! Return the <mean, std error> for the given vector of values.
  template <typename VectorType>
  std::pair<typename VectorType::value_type, typename VectorType::value_type>
  mean_stderr(const VectorType& vals) {
    typedef typename VectorType::value_type value_type;
    value_type sum(0);
    value_type sum2(0);
    if (vals.size() == 0)
      return std::make_pair(0, 0);
    foreach(value_type d, vals) {
      if (!is_finite(d))
        return std::make_pair(d,std::numeric_limits<value_type>::infinity());
      sum += d;
      sum2 += d * d;
    }
    value_type mean(sum / vals.size());
    value_type stderror(std::sqrt(sum2 / vals.size() - mean * mean)
                    / std::sqrt(vals.size()));
    if (std::isnan(stderror)) {
      // Check for numerical issues.
      value_type tmp = sum2 / vals.size() - mean * mean;
      if (tmp < 0  &&  fabs(tmp) < 0.000000001)
        stderror = 0;
    }
    return std::make_pair(mean, stderror);
  }

  //! Return the <median, Median Absolute Deviation> for the given vector of
  //! values.
  template <typename VectorType>
  std::pair<typename VectorType::value_type, typename VectorType::value_type>
  median_MAD(const VectorType& vals) {
    typedef typename VectorType::value_type value_type;
    assert(vals.size() != 0);
    std::vector<value_type> sorted(vals.begin(), vals.end());
    std::sort(sorted.begin(), sorted.end());
    size_t median_i(sorted.size() / 2);
    value_type median(sorted[median_i]);
    foreach(value_type& v, sorted)
      v = fabs(v - median);
    std::sort(sorted.begin(), sorted.end());
    return std::make_pair(median, sorted[median_i]);
  }

  namespace statistics {

    enum generalized_mean_enum { MEAN, MEDIAN };

    //! Returns name of the given generalized mean as a string.
    std::string generalized_mean_string(generalized_mean_enum gm);

    //! Returns name of the given generalized mean's corresponding deviation
    //! as a string.
    std::string generalized_deviation_string(generalized_mean_enum gm);

  } // namespace statistics

  oarchive& operator<<(oarchive& a, statistics::generalized_mean_enum gm);

  iarchive& operator>>(iarchive& a, statistics::generalized_mean_enum& gm);

  //! Return the generalized mean of type GM of the given values.
  template <typename T>
  T generalized_mean(const arma::Col<T>& v,
                     statistics::generalized_mean_enum gm) {
    switch (gm) {
    case statistics::MEAN:
      return mean(v);
    case statistics::MEDIAN:
      return median(v);
    default:
      throw std::invalid_argument("generalized_mean(v,gm) given bad gm value.");
    }
  }

  //! Return the generalized mean of type GM of the given values.
  template <typename T>
  T generalized_mean(const std::vector<T>& v,
                     statistics::generalized_mean_enum gm) {
    switch (gm) {
    case statistics::MEAN:
      return mean(v);
    case statistics::MEDIAN:
      return median(v);
    default:
      throw std::invalid_argument("generalized_mean(v,gm) given bad gm value.");
    }
  }

  //! Return the generalized deviation of type GM of the given values.
  template <typename T>
  T generalized_deviation(const arma::Col<T>& v,
                          statistics::generalized_mean_enum gm) {
    switch (gm) {
    case statistics::MEAN:
      return mean_stderr(v).second;
    case statistics::MEDIAN:
      return median_MAD(v).second;
    default:
      throw std::invalid_argument
        ("generalized_deviation(v,gm) given bad gm value.");
    }
  }

  //! Return the generalized deviation of type GM of the given values.
  template <typename T>
  T generalized_deviation(const std::vector<T>& v,
                          statistics::generalized_mean_enum gm) {
    switch (gm) {
    case statistics::MEAN:
      return mean_stderr(v).second;
    case statistics::MEDIAN:
      return median_MAD(v).second;
    default:
      throw std::invalid_argument
        ("generalized_deviation(v,gm) given bad gm value.");
    }
  }

  /**
   * Lightweight accumulator for computing mean, stderr, min, max.
   */
  template <typename T>
  class LightStatsAccumulator {

  public:
    LightStatsAccumulator()
      : sum(0), sum2(0), min_val(0), max_val(0), cnt(0) { }

    void save(oarchive& ar) const {
      ar << sum << sum2 << min_val << max_val << cnt;
    }

    void load(iarchive& ar) {
      ar >> sum >> sum2 >> min_val >> max_val >> cnt;
    }

    //! Add a value to the accumulator.
    void push(T val) {
      sum += val;
      sum2 += sqr(val);
      if (cnt == 0) {
        min_val = val;
        max_val = val;
      } else {
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
      }
      ++cnt;
    }

    //! Return mean, or 0 if no items have been added.
    T mean() const {
      if (cnt == 0) return 0;
      return (sum / cnt);
    }

    //! Return standard error of the mean, or 0 if no items have been added.
    //! The standard deviation of the sample is normalized by N (count),
    //! not by N-1.
    T stderror() const {
      if (cnt == 0) return 0;
      return std::sqrt(((sum2 / cnt) - sqr(mean())) / cnt);
    }

    //! Return min value,
    //! or -std::numeric_limits<T>::max() if no items have been added.
    T min() const {
      if (cnt == 0) return -std::numeric_limits<T>::max();
      return min_val;
    }

    //! Return max value,
    //! or std::numeric_limits<T>::max() if no items have been added.
    T max() const {
      if (cnt == 0) return std::numeric_limits<T>::max();
      return max_val;
    }

    //! Number of items added.
    size_t count() const { return cnt; }

  protected:
    //! Sum of values added
    T sum;
    //! Sum of squares of values added
    T sum2;
    //! Min value
    T min_val;
    //! Max value
    T max_val;
    //! Count of values added
    size_t cnt;

  }; // class LightStatsAccumulator

  // Max and min: deterministic tie-breaking
  //============================================================================

  /**
   * Return the index of an extreme value in the vector,
   * as specified by the template parameter Comparator.
   * If multiple values are extreme, choose the first one.
   *
   * @tparam Comparator  This defines the extremum.
   *                     If Comparator(a,b) == false, then a is more extreme.
   *                     E.g., std::less_equal<T> (<=) yields the maximum.
   *                     This type must be default constructable.
   * @param v   Vector of values (of size > 0).
   */
  template <typename T, typename Comparator>
  size_t extreme_index(const forward_range<T>& v) {
    typename forward_range<T>::const_iterator it(boost::begin(v));
    typename forward_range<T>::const_iterator end(boost::end(v));
    assert(it != end);
    Comparator comp;
    T best(*it);
    ++it;
    size_t best_index = 0;
    size_t i = 1;
    while (it != end) {
      if (!comp(*it, best)) {
        best = *it;
        best_index = i;
      }
      ++it;
      ++i;
    }
    return best_index;
  }

  //! Return the index of the max value in the vector.
  //! If multiple values are maximal, choose the first one.
  template <typename T>
  size_t max_index(const forward_range<T>& v) {
    return extreme_index<T, std::less_equal<T> >(v);
  }

  // For std::vector
  template <typename T>
  size_t max_index(const std::vector<T>& v) {
    return extreme_index<T, std::less_equal<T> >(v);
  }

  //! For arma::Col
  template <typename T>
  size_t max_index(const arma::Col<T>& v) {
    return extreme_index<T, std::less_equal<T> >(v);
  }

  //! Return the index of the min value in the vector.
  //! If multiple values are minimal, choose the first one.
  template <typename T>
  size_t min_index(const forward_range<T>& v) {
    return extreme_index<T, std::greater_equal<T> >(v);
  }

  // For std::vector
  template <typename T>
  size_t min_index(const std::vector<T>& v) {
    return extreme_index<T, std::greater_equal<T> >(v);
  }

  // For arma::Col
  template <typename T>
  size_t min_index(const arma::Col<T>& v) {
    return extreme_index<T, std::greater_equal<T> >(v);
  }

  //! Return the max value in the vector.
  //! If multiple values are maximal, choose the first one.
  template <typename T>
  T max(const std::vector<T>& v) {
    return v[extreme_index<T, std::less_equal<T> >(v)];
  }

  //! Return the min value in the vector.
  //! If multiple values are minimal, choose the first one.
  template <typename T>
  T min(const std::vector<T>& v) {
    return v[extreme_index<T, std::greater_equal<T> >(v)];
  }

  // Max and min: random tie-breaking
  //============================================================================

  /**
   * Return the index of an extreme value in the vector,
   * as specified by the template parameter Comparator.
   * If multiple values are extreme, choose one with uniform probability.
   *
   * @tparam Comparator  This defines the extremum.
   *                     If Comparator(a,b) == false, then a is more extreme.
   *                     It should not be the case that Comparator(a,b) and
   *                     Comparator(b,a) both hold; i.e., this can be < but
   *                     not <=.
   *                     E.g., std::less<T> (<=) yields the maximum.
   *                     This type must be default constructable.
   * @tparam Engine      Random number generator.
   * @param v   Vector of values (of size > 0).
   */
  template <typename T, typename Comparator, typename Engine>
  size_t extreme_index(const forward_range<T>& v, Engine& rng) {
    boost::uniform_real<double> uniform_prob(0,1);
    typename forward_range<T>::const_iterator it(boost::begin(v));
    typename forward_range<T>::const_iterator end(boost::end(v));
    assert(it != end);
    Comparator comp;
    T best(*it);
    ++it;
    size_t nbest(1);
    size_t best_index(0);
    size_t i(1);
    while (it != end) {
      if (comp(*it, best)) {
        ++it;
        ++i;
        continue;
      }
      if (*it == best) {
        ++nbest;
        if (uniform_prob(rng) > 1. / nbest) {
          ++it;
          ++i;
          continue;
        }
      } else
        nbest = 1;
      best = *it;
      best_index = i;
      ++it;
      ++i;
    }
    return best_index;
  }

  //! Return the index of the max value in the vector.
  //! If multiple values are maximal, choose one with uniform probability.
  template <typename T, typename Engine>
  size_t max_index(const forward_range<T>& v, Engine& rng) {
    return extreme_index<T, std::less<T>, Engine>(v, rng);
  }

  // For std::vector
  template <typename T, typename Engine>
  size_t max_index(const std::vector<T>& v, Engine& rng) {
    return extreme_index<T, std::less<T>, Engine>(v, rng);
  }

  // For sill::vector
  template <typename T, typename Engine>
  size_t max_index(const arma::Col<T>& v, Engine& rng) {
    return extreme_index<T, std::less<T>, Engine>(v, rng);
  }

  //! Return the index of the min value in the vector.
  //! If multiple values are minimal, choose one with uniform probability.
  template <typename T, typename Engine>
  size_t min_index(const forward_range<T>& v, Engine& rng) {
    return extreme_index<T, std::greater<T>, Engine>(v, rng);
  }

  // For std::vector
  template <typename T, typename Engine>
  size_t min_index(const std::vector<T>& v, Engine& rng) {
    return extreme_index<T, std::greater<T>, Engine>(v, rng);
  }

  // For sill::vector
  template <typename T, typename Engine>
  size_t min_index(const arma::Col<T>& v, Engine& rng) {
    return extreme_index<T, std::greater<T>, Engine>(v, rng);
  }

  //! Return the indices of the max value in the matrix.
  //! If multiple values are maximal, choose one with uniform probability.
  template <typename Engine>
  std::pair<size_t,size_t> max_indices(const mat& m, Engine& rng) {
    boost::uniform_real<double> uniform_prob(0,1);
    if (m.size() == 0)
      return std::make_pair(0,0);
    double best(m(0,0));
    size_t nbest(0);
    std::pair<size_t,size_t> best_indices(std::make_pair(0,0));
    for (size_t i(0); i < m.n_rows; ++i) {
      for (size_t j(0); j < m.n_cols; ++j) {
        if (m(i,j) < best)
          continue;
        if (m(i,j) == best) {
          ++nbest;
          if (uniform_prob(rng) > 1. / nbest)
            continue;
        } else {
          nbest = 1;
        }
        best = m(i,j);
        best_indices.first = i;
        best_indices.second = j;
      }
    }
    return best_indices;
  }

  // Order statistics
  //============================================================================

  namespace impl {

    //! Used for function sorted_indices(vec)
    template <typename VecType>
    class sorted_indices_comparator {
      const VecType& v;
    public:
      explicit sorted_indices_comparator(const VecType& v) : v(v) { }
      bool operator()(size_t a, size_t b) const { return (v[a] < v[b]); }
    }; // class sorted_indices_comparator

    //! Used for function sorted_indices(std::vector<vec>)
    class sorted_indices_comparator2 {
      const std::vector<vec>& v;
    public:
      explicit sorted_indices_comparator2(const std::vector<vec>& v) : v(v) { }
      bool operator()(size_t a, size_t b) const {
        size_t min_size = std::min<size_t>(v[a].size(), v[b].size());
        for (size_t i = 0; i < min_size; ++i) {
          if (v[a][i] != v[b][i])
            return (v[a][i] < v[b][i]);
        }
        return (v[a].size() < v[b].size());
      }
    }; // class sorted_indices_comparator2

  } // namespace impl

  //! Given a vector of values, return a list of indices which give the values
  //! in increasing order.
  template <typename VecType>
  std::vector<size_t> sorted_indices(const VecType& v) {
    std::vector<size_t> ind(v.size());
    for (size_t i(0); i < v.size(); ++i)
      ind[i] = i;
    impl::sorted_indices_comparator<VecType> comp(v);
    std::sort(ind.begin(), ind.end(), comp);
    return ind;
  }

  //! Given a vector of values, return a list of indices which give the values
  //! in increasing order.
  template <typename VecType, typename IndexVecType>
  IndexVecType sorted_indices(const VecType& v) {
    IndexVecType ind(v.size());
    for (size_t i(0); i < v.size(); ++i)
      ind[i] = i;
    impl::sorted_indices_comparator<VecType> comp(v);
    std::sort(ind.begin(), ind.end(), comp);
    return ind;
  }

  //! Given a vector of vecs, return a list of indices which give the vecs
  //! in lexigraphical order (increasing).
  std::vector<size_t> sorted_indices(const std::vector<vec>& v);

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_MATH_STATISTICS_HPP

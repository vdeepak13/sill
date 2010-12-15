
#ifndef PRL_MATH_STATISTICS_HPP
#define PRL_MATH_STATISTICS_HPP

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>

#include <prl/math/linear_algebra.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  // Means, standard errors, medians, and MADs
  //============================================================================

  //! Return the <mean, std error> for the given vector of values.
  std::pair<double, double> mean_stderr(const std::vector<double>& vals);

  //! Return the <mean, std error> for the given vector of values.
  std::pair<double, double> mean_stderr(const vec& vals);

  //! Return the <median, Median Absolute Deviation> for the given vector of
  //! values.
  std::pair<double, double> median_MAD(const std::vector<double>& vals);

  //! Return the <median, Median Absolute Deviation> for the given vector of
  //! values.
  std::pair<double, double> median_MAD(const vec& vals);

  // Max and min
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
    size_t best_index(0);
    size_t i(1);
    while (it != end) {
      if (comp(*it, best)) {
        ++it;
        ++i;
        continue;
      }
      best = *it;
      best_index = i;
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

  //! Return the index of the min value in the vector.
  //! If multiple values are minimal, choose the first one.
  template <typename T>
  size_t min_index(const forward_range<T>& v) {
    return extreme_index<T, std::greater_equal<T> >(v);
  }

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

  // For prl::vector
  template <typename T, typename Engine>
  size_t max_index(const vector<T>& v, Engine& rng) {
    return extreme_index<T, std::less<T>, Engine>(v, rng);
  }

  // For itpp::vec
  template <typename T, typename Engine>
  size_t max_index(const itpp::Vec<T>& v, Engine& rng) {
    return extreme_index<T, std::less<T>, Engine>(vector<T>(v), rng);
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

  // For prl::vector
  template <typename T, typename Engine>
  size_t min_index(const vector<T>& v, Engine& rng) {
    return extreme_index<T, std::greater<T>, Engine>(v, rng);
  }

  // For itpp::vec
  template <typename T, typename Engine>
  size_t min_index(const itpp::Vec<T>& v, Engine& rng) {
    return extreme_index<T, std::greater<T>, Engine>(vector<T>(v), rng);
  }

  //! Return the indices of the max value in the matrix.
  //! If multiple values are maximal, choose one with uniform probability.
  template <typename Engine>
  std::pair<size_t,size_t> max_indices(const mat& v, Engine& rng) {
    boost::uniform_real<double> uniform_prob(0,1);
    if (v.size() == 0)
      return std::make_pair(0,0);
    double best(v(0,0));
    size_t nbest(0);
    std::pair<size_t,size_t> best_indices(std::make_pair(0,0));
    for (size_t i(0); i < v.size1(); ++i) {
      for (size_t j(0); j < v.size2(); ++j) {
        if (v(i,j) < best)
          continue;
        if (v(i,j) == best) {
          ++nbest;
          if (uniform_prob(rng) > 1. / nbest)
            continue;
        } else {
          nbest = 1;
        }
        best = v(i,j);
        best_indices.first = i;
        best_indices.second = j;
      }
    }
    return best_indices;
  }

  // Order statistics
  //============================================================================

  namespace impl {
    //! Used for function sorted_indices()
    class sorted_indices_comparator {
      const vec& v;
    public:
      explicit sorted_indices_comparator(const vec& v) : v(v) { }
      bool operator()(size_t a, size_t b) const { return (v[a] < v[b]); }
    }; // class sorted_indices_comparator
  } // namespace impl

  //! Given a vector of values, return a list of indices which give the values
  //! in sorted order.
  //! @todo Once paraml has been merged back with the trunk and we have
  //!       decided how to handle ranges, put this somewhere else.
  std::vector<size_t> sorted_indices(const vec& v);

} // end of namespace: prl

#include <prl/macros_undef.hpp>

#endif // #ifndef PRL_MATH_STATISTICS_HPP

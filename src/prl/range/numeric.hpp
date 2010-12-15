#ifndef PRL_NUMERIC_HPP
#define PRL_NUMERIC_HPP

#include <numeric>

#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>

#include <prl/global.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  //! \addtogroup range_numeric
  //! @{

  template <class InputRange, class T>
  T accumulate(const InputRange& range, T init) {
    return std::accumulate(boost::begin(range), boost::end(range), init);
  }


  template <class InputRange, class T, class BinaryOperation>
  T accumulate(const InputRange& range, T init, BinaryOperation op) {
    return std::accumulate(boost::begin(range), boost::end(range), init, op);
  }

  template <class InputRange1, class InputRange2, class T>
  T inner_product(const InputRange1& range1, const InputRange2& range2, T init){
    return std::inner_product(boost::begin(range1), boost::end(range1),
                              boost::begin(range2),
                              init);
  }
  
  template<class InputRange1, class InputRange2,
           class T,
           class BinaryOperation1, class BinaryOperation2>
  T inner_product(const InputRange1& range1,
                  const InputRange2& range2,
                  T init,
                  BinaryOperation1 op1,
                  BinaryOperation2 op2) {
    return std::inner_product(boost::begin(range1), boost::end(range1),
                              boost::begin(range2),
                              init, op1, op2);
  }

  template <class InputRange, class OutputIterator>
  OutputIterator partial_sum(const InputRange& range, OutputIterator result) {
    return std::partial_sum(boost::begin(range), boost::end(range), result);
  }
  
  template <class InputRange, class OutputIterator,
            class BinaryOperation>
  OutputIterator partial_sum(const InputRange& range, OutputIterator result,
                             BinaryOperation op) {
    return std::partial_sum(boost::begin(range), boost::end(range), result, op);
  }

  template <class InputRange, class OutputIterator>
  OutputIterator adjacent_difference(const InputRange& range,
                                     OutputIterator result) {
    return std::adjacent_difference(boost::begin(range), boost::end(range),
                                    result);
  }
  
  template <class InputRange, class OutputIterator,
            class BinaryOperation>
  OutputIterator adjacent_difference(const InputRange& range,
                                     OutputIterator result,
                                     BinaryOperation op) {
    return std::adjacent_difference(boost::begin(range), boost::end(range),
                                    result, op);
  }

  //! @} group range_numeric

} // namespace prl

#include <prl/macros_undef.hpp>

#endif



#if 0
  // The following functions are deprecated.
  // Use matrix-specific functions instead

  /**
   * Computes the sum for a collection of values.  If the collection is
   * empty, returns static_cast<T>(0) where T is the range value type.
   */
  template <typename R>
  typename R::value_type sum(const R& values) {
    concept_assert((InputRange<R>));
    typedef typename R::value_type T;
    if (boost::empty(values)) 
      return T();
    else
      return prl::accumulate(values  | prl::dropped(1),
			       prl::front(values), std::plus<T>());
  }

  /**
   * Computes the product for a collection of values. If the collection is
   * empty, returns static_cast<T>(1) where T is the range value type.
   */
  template <typename R>
  typename R::value_type prod(const R& values) {
    concept_assert((InputRange<R>));
    typedef typename R::value_type T;
    if (values.empty()) 
      return T(1);
    else
      return prl::accumulate(values  | prl::dropped(1),
			       prl::front(values), std::multiplies<T>());
  }

  //! Computes the mean of a collection of values
  template <typename R>
  typename R::value_type mean(const R& values) {
    concept_assert((ReadableForwardRange<R>));
    typedef typename R::value_type T;
    if (values.empty())
      return static_cast<T>(0);
    else
      return sum(values) / values.size();
  }

  /**
   * Computes the variance for a collection of values.
   * If unbiased = true, normalizes by N-1; otherwise normalizes by N.
   */
  template <typename R>
  typename R::value_type var(const R& values, bool unbiased = true) {
    concept_assert((ReadableForwardRange<R>));
    assert(!values.empty());
    typedef typename R::value_type T;
    T sum_squares = sum(make_transformed(values, squared<T>()));
    return sum_squares / (values.size()-unbiased) - sqr(mean(values));
  }

  /**
   * Computes the covariance between two collections of values.
   * If unbiased = true, normalizes by N-1; otherwise, normalizes by N.
   * @see cov<M,R> for vectorized version
   */
  template <typename R>
  typename R::value_type cov(const R& x, const R& y, bool unbiased = true) {
    concept_assert((ReadableForwardRange<R>));
    typedef typename R::value_type T;
    //typedef tuple<T, T> ref_pair;
    //typedef tuple<const T&, const T&> ref_pair;
    assert(!x.empty() && x.size()==y.size());

    T sum_squares = 0;
    //foreach(ref_pair t, make_tuple(x, y) | prl::zipped) t.get(0);
    foreach_auto(t, make_tuple(ref(x), ref(y)) | prl::zipped) 
      // Warning: if we did not use ref() or tuple_xy here, the temporary
      //          R objects could go out of scope & get deleted before
      //          we access them.
      sum_squares += boost::get<0>(t) * boost::get<1>(t);

    return sum_squares / (x.size()-unbiased) - mean(x) * mean(y);
  }


  /**********************************************************
   * Functions on ranges of vectors.
   **********************************************************/

  /**
   * Computes the covariance for a collection of vectors.
   * @param unbiased, if true, normalizes by N-1; otherwise, normalizes by N
   */
  template <typename M, typename R>
  M cov(const R& vectors, bool unbiased = true) {
    typedef typename R::value_type V;
    concept_assert((ReadableForwardRange<R>));
    concept_assert((Mutable_Matrix<M>));
    concept_assert((Vector<V>));
    assert(!vectors.empty());
    
    V mu = mean(vectors);
    M result = zeros(mu.size(), mu.size());
    foreach(const V& v, vectors)
      result += outer_prod(v, v);
    return (result / (vectors.size()-unbiased)) - outer_prod(mu, mu);
  }

  /**
   * Computes the cross-covariance between two collections of vectors
   * @param unbiased, if true, normalizes by N-1; otherwise, normalizes by N
   */
  template <typename M, typename R>
  M cov(const R& x, const R& y, bool unbiased = true) {
    typedef typename R::value_type V;
    concept_assert((ReadableForwardRange<R>));
    concept_assert((Mutable_Matrix<M>));
    concept_assert((Vector<V>));
    assert(!x.empty());
    assert(x.size() == y.size());

    V mx = mean(x), my = mean(y);
    M result = zeros(mx.size(), my.size());
    // tuple<const R&, const R&> tuple_xy(x,y);
    // tuple<R,R> would also work but would create a copy
    // foreach_auto(t, tuple_xy | prl::zipped) {
    foreach_auto(t, make_tuple(ref(x), ref(y)) | prl::zipped) {
      // Warning: if we did not use ref() or tuple_xy here, the temporary
      //          R objects could go out of scope & get deleted before
      //          we access them.
      result += outer_prod(boost::get<0>(t) - mx, boost::get<1>(t) - my);
    }
    return (result / (x.size()-unbiased));// - outer_prod(mx, my);
  }

#endif

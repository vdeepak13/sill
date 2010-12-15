#ifndef PRL_ALGORITHM_HPP
#define PRL_ALGORITHM_HPP

#include <algorithm>
#include <functional>

#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/empty.hpp>
#include <boost/range/iterator.hpp>
#include <boost/range/difference_type.hpp>
#include <boost/range/value_type.hpp>
#include <boost/type_traits/remove_const.hpp>

#include <prl/functional.hpp> // minimum & maximum
#include <prl/range/concepts.hpp>
#include <prl/stl_concepts.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  //! \addtogroup range_algorithm
  //! @{

  // Nonmodifying sequence operations
  //============================================================================
  template<class Range, class Function>
  Function for_each(const Range& range, Function f) {
    return std::for_each(boost::begin(range), boost::end(range), f);
  }
  
  template<class Range, class T>
  typename boost::range_iterator<Range>::type
  find(const Range& range, const T& value) {
    return std::find(boost::begin(range), boost::end(range), value);
  }
  
  template<class Range, class Predicate>
  typename boost::range_iterator<Range>::type
  find_if(const Range& range, Predicate pred) {
    return std::find_if(boost::begin(range), boost::end(range), pred);
  }
  
  template<class Range1, class Range2>
  typename boost::range_iterator<Range1>::type 
  find_end(const Range1& range1, const Range2& range2) {
    return std::find_end(boost::begin(range1), boost::end(range1),
                         boost::begin(range2), boost::end(range2));
  }
  
  template<class Range1, class Range2, class BinaryPredicate>
  typename boost::range_iterator<Range1>::type
  find_end(const Range1& range1, const Range2& range2, BinaryPredicate pred) {
    return std::find_end(boost::begin(range1), boost::end(range1),
                         boost::begin(range2), boost::end(range2), 
                         pred);
  }
  
  template<class Range1, class Range2>
  typename boost::range_iterator<Range1>::type
  find_first_of(const Range1& range1, const Range2& range2) {
    return std::find_first_of(boost::begin(range1), boost::end(range1),
                              boost::begin(range2), boost::end(range2));
  }

  template<class Range1, class Range2, class BinaryPredicate>
  typename boost::range_iterator<Range1>::type
  find_first_of(const Range1& range1, const Range2& range2, BinaryPredicate p) {
    return std::find_first_of(boost::begin(range1), boost::end(range1),
                              boost::begin(range2), boost::end(range2), 
                              p);
  }

  template<class Range>
  typename boost::range_iterator<Range>::type
  adjacent_find(const Range& range) {
    return std::adjacent_find(boost::begin(range), boost::end(range));
  }
  
  template<class Range, class BinaryPredicate>
  typename boost::range_iterator<Range>::type
  adjacent_find(const Range& range, BinaryPredicate pred) {
    return std::adjacent_find(boost::begin(range), boost::end(range), pred);
  }

  template<class Range, class T>
  typename boost::range_difference<Range>::type
  count(const Range& range, const T& value) {
    return std::count(boost::begin(range), boost::end(range), value);
  }

  template<class Range, class Predicate>
  typename boost::range_difference<Range>::type
  count_if(const Range& range, Predicate pred) {
    return std::count_if(boost::begin(range), boost::end(range), pred);
  }
  
  template<class Range1, class Range2>
  std::pair<typename boost::range_iterator<Range1>::type,
            typename boost::range_iterator<Range2>::type>
  mismatch(const Range1& range1, const Range2& range2) {
    return std::mismatch(boost::begin(range1), boost::end(range1),
                         boost::begin(range2), boost::end(range2));
  }
  
  template<class Range1, class Range2, 
           class BinaryPredicate>
  std::pair<typename boost::range_iterator<Range1>::type,
            typename boost::range_iterator<Range2>::type>
  mismatch(const Range1& range1, const Range2& range2, BinaryPredicate pred) {
    return std::mismatch(boost::begin(range1), boost::end(range1),
                         boost::begin(range2), boost::end(range2), pred);
  }
  
  template<class Range1, class Range2>
  bool equal(const Range1& range1, const Range2& range2) {
    return std::equal(boost::begin(range1), boost::end(range1),
                      boost::begin(range2));
  }
   
  template<class Range1, class Range2, class BinaryPredicate>
  bool equal(const Range1& range1, const Range2& range2, BinaryPredicate pred) {
    return std::equal(boost::begin(range1), boost::end(range1),
                      boost::begin(range2), 
                      pred);
  }
  
  template<class Range1, class Range2>
  typename boost::range_iterator<Range1>::type 
  search(const Range1& range1, const Range2& range2) {
    return std::search(boost::begin(range1), boost::end(range1),
                       boost::begin(range2), boost::end(range2));
  }

  template<class Range1, class Range2, class BinaryPredicate>
  typename boost::range_iterator<Range1>::type 
  search(const Range1& range1, const Range2& range2, BinaryPredicate pred) {
    return std::search(boost::begin(range1), boost::end(range1),
                       boost::begin(range2), boost::end(range2), pred);
  }
  
  template<class Range, class Size, class T>
  typename boost::range_iterator<Range>::type
  search_n(const Range& range, Size count, const T& value) {
    return std::search_n(boost::begin(range), boost::end(range),
                         count, value);
  }

  template<class Range, class Size, class T, class BinaryPredicate>
  typename boost::range_iterator<Range>::type
  search_n(const Range& range, Size count, const T& value, BinaryPredicate p) {
    return std::search_n(boost::begin(range), boost::end(range),
                         count, value, p);
  }

  // Modifying sequence operations
  //============================================================================
  template<class Range, class Function>
  Function for_each(Range& range, Function f) {
    return std::for_each(boost::begin(range), boost::end(range), f);
  }

  // Copies
  //============================================================================
  template<class Range, class OutputIterator>
  OutputIterator copy(const Range& range, OutputIterator result) {
    return std::copy(boost::begin(range), boost::end(range), result);
  }
  
  template<class Range, class OutputIterator>
  OutputIterator copy_backward(const Range& range, OutputIterator result) {
    return std::copy_backward(boost::begin(range), boost::end(range), result);
  }
  
  // Swaps
  //============================================================================
  template<class Range1, class Range2>
  typename boost::range_iterator<Range2>::type 
  swap_ranges(Range1& range1, Range2& range2) {
    return std::swap_ranges(boost::begin(range1), boost::end(range1),
                            boost::begin(range2), boost::end(range2));
  }
  
  template<class Range, class OutputIterator, class UnaryOperation>
  OutputIterator transform(const Range& range, 
                           OutputIterator result, 
                           UnaryOperation op) {
    return std::transform(boost::begin(range), boost::end(range), result, op);
  }
  
  template<class Range1, 
           class Range2, 
           class OutputIterator,
           class BinaryOperation>
  OutputIterator transform(const Range1& range1, 
                           const Range2& range2, 
                           OutputIterator result,
                           BinaryOperation binary_op) {
    return std::transform(boost::begin(range1), boost::end(range1), 
                          boost::begin(range2), boost::end(range2), 
                          result, binary_op);
  }
  
  template<class Range, class T>
  void replace(const Range& range, const T& old_value, const T& new_value) {
    std::replace(boost::begin(range), boost::end(range), old_value, new_value);
  }
  
  template<class Range, class Predicate, class T>
  void replace_if(const Range& range, Predicate pred, const T& new_value) {
    std::replace_if(boost::begin(range), boost::end(range), pred, new_value);
  }

  template<class Range, class OutputIterator, class T>
  OutputIterator replace_copy(const Range& range, 
                              OutputIterator result,
                              const T& old_value, 
                              const T& new_value);
  
  template<class Range, class OutputIterator, 
           class Predicate, class T>
  OutputIterator replace_copy_if(const Range& range, 
                                 OutputIterator result,
                                 Predicate pred,
                                 const T& new_value);
  
  template<class Range, class T>
  void fill(Range& range, const T& value) {
    std::fill(boost::begin(range), boost::end(range), value);
  }
  
  template<class Range, class Generator>
  void generate(Range& range, Generator gen);
  
  template<class Range, class T>
  typename boost::range_iterator<Range>::type
  remove(Range& range, const T& value);
  
  template<class Range, class Predicate>
  typename boost::range_iterator<Range>::type
  remove_if(Range& range, Predicate pred);
  
  template<class Range, class OutputIterator, class T>
  OutputIterator remove_copy(const Range& range, 
                             OutputIterator result, 
                             const T& value);
  
  template<class Range, class OutputIterator, class Predicate>
  OutputIterator remove_copy_if(const Range& range,
                                OutputIterator result, 
                                Predicate pred);
  
  //! range-based version of copy_if
  template<typename InputRange, typename OutputIterator, typename Predicate>
  OutputIterator copy_if(const InputRange& range,
			 OutputIterator out, Predicate pred) {
    typedef typename boost::range_iterator<const InputRange>::type iterator;
    iterator it = boost::begin(range), end = boost::end(range);
    while(it != end) {
      if (pred(*it)) {
        *out = *it;
        ++out;
      }
      ++it;
    }
    return out;
  }

  template<class Range>
  typename boost::range_iterator<Range>::type unique(Range& range);
  
  template<class Range, class BinaryPredicate>
  typename boost::range_iterator<Range>::type unique(Range& range, 
                                                     BinaryPredicate pred);
  
  template<class Range, class OutputIterator>
  OutputIterator unique_copy(const Range& range, 
                             OutputIterator result);
  
  template<class Range, class OutputIterator, 
           class BinaryPredicate>
  OutputIterator unique_copy(const Range& range, 
                             OutputIterator result,
                             BinaryPredicate binary_pred);
                             
  template<class Range>
  void reverse(Range& range);
  
  template<class Range, class OutputIterator>
  OutputIterator reverse_copy(const Range& range,
                              OutputIterator result);
  
  template<class RandomAccessRange>
  void random_shuffle(RandomAccessRange& start);
  
  template<class RandomAccessRange, 
           class RandomNumberGenerator>
  void random_shuffle(RandomAccessRange& range, RandomNumberGenerator& rand);
  
  // Partitions
  //============================================================================
  template<class Range, class Predicate>
  typename boost::range_iterator<Range>::type 
  partition(Range& range, Predicate pred) {
    return std::partition(boost::begin(range), boost::end(range), pred);
  }
  
  template<class Range, class Predicate>
  typename boost::range_iterator<Range>::type 
  stable_partition(Range& range, Predicate pred) {
    return std::stable_partition(boost::begin(range), boost::end(range), pred);
  }
  
  // Sorting
  //============================================================================
  template<class RandomAccessRange>
  void sort(RandomAccessRange& range) {
    std::sort(boost::begin(range), boost::end(range));
  }
  
  template<class RandomAccessRange, class Compare>
  void sort(RandomAccessRange& range, Compare comp) {
    std::sort(boost::begin(range), boost::end(range), comp);
  }
  
  template<class RandomAccessRange>
  void stable_sort(RandomAccessRange& range) {
    std::stable_sort(boost::begin(range), boost::end(range));
  }
  
  template<class RandomAccessRange, class Compare>
  void stable_sort(RandomAccessRange& range, Compare comp) {
    std::stable_sort(boost::begin(range), boost::end(range), comp);
  }
  
  // Binary search
  //============================================================================
  template<class Range, class T>
  typename boost::range_iterator<Range>::type 
  lower_bound(const Range& range, const T& value);
  
  template<class Range, class T, class Compare>
  typename boost::range_iterator<Range>::type
  lower_bound(const Range& range, const T& value, Compare comp);
  
  template<class Range, class T>
  typename boost::range_iterator<Range>::type 
  upper_bound(const Range& range, const T& value);
  
  template<class Range, class T, class Compare>
  typename boost::range_iterator<Range>::type
  upper_bound(const Range& range, const T& value, Compare comp);
  
  template<class Range, class T>
  std::pair<typename boost::range_iterator<Range>::type,
            typename boost::range_iterator<Range>::type>
  equal_range(const Range& range, const T& value);
  
  template<class Range, class T, class Compare>
  std::pair<typename boost::range_iterator<Range>::type,
            typename boost::range_iterator<Range>::type>
  equal_range(const Range& range, const T& value, Compare comp);
  
  template<class RandomAccessRange, class T>
  bool binary_search(const RandomAccessRange& range, 
                     const T& value);
  
  template<class RandomAccessRange, class T, class Compare>
  bool binary_search(const RandomAccessRange& range, 
                     const T& value, 
                     Compare comp);
  
  // Merging
  //============================================================================
  template<class Range1, class Range2, 
           class OutputIterator>
  OutputIterator merge(const Range1& range1, 
                       const Range2& range2, 
                       OutputIterator result) {
    return std::merge(boost::begin(range1), boost::end(range1),
                      boost::begin(range2), boost::end(range2),
                      result);
  }
  
  template<class Range1, class Range2,
           class OutputIterator, class Compare>
  OutputIterator merge(const Range1& range1, 
                       const Range2& range2, 
                       OutputIterator result, 
                       Compare comp) {
    return std::merge(boost::begin(range1), boost::end(range1),
                      boost::begin(range2), boost::end(range2),
                      result, comp);
  }  

  // Set operations
  //============================================================================
  template<class Range1, class Range2>
  bool includes(const Range1& range1,
                const Range2& range2) {
    return std::includes(boost::begin(range1), boost::end(range1),
                         boost::begin(range2), boost::end(range2));
  }
  
  template<class Range1, class Range2, class Compare>
  bool includes(const Range1& range1, 
                const Range2& range2, 
                Compare comp) {
    return std::includes(boost::begin(range1), boost::end(range1),
                         boost::begin(range2), boost::end(range2),
                         comp);
  }
  
  template<class Range1, class Range2, class OutputIterator>
  OutputIterator set_union(const Range1& range1, 
                           const Range2& range2, 
                           OutputIterator result) {
    return std::set_union(boost::begin(range1), boost::end(range1),
                          boost::begin(range2), boost::end(range2),
                          result);
  }
  
  template<class Range1, class Range2, class OutputIterator, class Compare>
  OutputIterator set_union(const Range1& range1, 
                           const Range2& range2, 
                           OutputIterator result, 
                           Compare comp) {
    return std::set_union(boost::begin(range1), boost::end(range1),
                          boost::begin(range2), boost::end(range2),
                          result, comp);
  }
  
  template<class Range1, class Range2, class OutputIterator>
  OutputIterator set_intersection(const Range1& range1,
                                  const Range2& range2,
                                  OutputIterator result) {
    return std::set_intersection(boost::begin(range1), boost::end(range1),
                                 boost::begin(range2), boost::end(range2),
                                 result);
  }
  
  template<class Range1, class Range2, class OutputIterator, class Compare>
  OutputIterator set_intersection(const Range1& range1,
                                  const Range2& range2,
                                  OutputIterator result, 
                                  Compare comp) {
    return std::set_intersection(boost::begin(range1), boost::end(range1),
                                 boost::begin(range2), boost::end(range2),
                                 result, comp);
  }
  
  template<class Range1, class Range2, class OutputIterator>
  OutputIterator set_difference(const Range1& range1,
                                const Range2& range2,
                                OutputIterator result) {
    return std::set_difference(boost::begin(range1), boost::end(range1),
                               boost::begin(range2), boost::end(range2),
                               result);
  }
  
  template<class Range1, class Range2, class OutputIterator, class Compare>
  OutputIterator set_difference(const Range1& range1,
                                const Range2& range2,
                                OutputIterator result, 
                                Compare comp) {
    return std::set_difference(boost::begin(range1), boost::end(range1),
                               boost::begin(range2), boost::end(range2),
                               result, comp);
  }
  
  template<class Range1, class Range2, class OutputIterator>
  OutputIterator set_symmetric_difference(const Range1& range1, 
                                          const Range2& range2,
                                          OutputIterator result) {
    return std::set_symmetric_difference
      (boost::begin(range1), boost::end(range1),
       boost::begin(range2), boost::end(range2),
       result);
  }
  
  template<class Range1, class Range2, class OutputIterator, class Compare>
  OutputIterator set_symmetric_difference(const Range1& range1, 
                                          const Range2& range2,
                                          OutputIterator result,
                                          Compare comp) {
    return std::set_symmetric_difference
      (boost::begin(range1), boost::end(range1),
       boost::begin(range2), boost::end(range2),
       result, comp);
  }
  
  // Heap operations
  //============================================================================
  template<class RandomAccessRange>
  void push_heap(RandomAccessRange& range) {
    std::push_heap(boost::begin(range), boost::end(range));
  }

  template<class RandomAccessRange, class Compare>
  void push_heap(RandomAccessRange& range, Compare comp) {
    std::push_heap(boost::begin(range), boost::end(range), comp);
  }
  
  template<class RandomAccessRange>
  void pop_heap(RandomAccessRange& range) {
    std::pop_heap(boost::begin(range), boost::end(range));
  }

  
  template<class RandomAccessRange, class Compare>
  void pop_heap(RandomAccessRange& range, Compare comp) {
    std::pop_heap(boost::begin(range), boost::end(range), comp);
  }
  
  template<class RandomAccessRange>
  void make_heap(RandomAccessRange& range) {
    std::make_heap(boost::begin(range), boost::end(range));
  }

  
  template<class RandomAccessRange, class Compare>
  void make_heap(RandomAccessRange& range, Compare comp) {
    std::make_heap(boost::begin(range), boost::end(range));
  }
  
  template<class RandomAccessRange>
  void sort_heap(RandomAccessRange& range) {
    std::sort_heap(boost::begin(range), boost::end(range));
  }
  
  template<class RandomAccessRange, class Compare>
  void sort_heap(RandomAccessRange& range, Compare comp) {
    std::sort_heap(boost::begin(range), boost::end(range), comp);
  }

  // Minimum and maximum
  //============================================================================
  template<class Range>
  typename boost::range_iterator<Range>::type 
  min_element(const Range& range) {
    return std::min_element(boost::begin(range), boost::end(range));
  }

  template<class Range, class Compare>
  typename boost::range_iterator<Range>::type 
  min_element(const Range& range, Compare comp) {
    return std::min_element(boost::begin(range), boost::end(range), comp);
  }
  
  template<class Range>
  typename boost::range_iterator<Range>::type
  max_element(const Range& range) {
    return std::max_element(boost::begin(range), boost::end(range));
  }
  
  template<class Range, class Compare>
  typename boost::range_iterator<Range>::type
  max_element(const Range& range, Compare comp) {
    return std::max_element(boost::begin(range), boost::end(range), comp);
  }
  
  template<class Range1, class Range2>
  bool lexicographical_compare(const Range1& range1,
                               const Range2& range2) {
    return std::lexicographical_compare
      (boost::begin(range1), boost::end(range1),
       boost::begin(range2), boost::end(range2));
  }
  
  template<class Range1, class Range2, class Compare>
  bool lexicographical_compare(const Range1& range1,
                               const Range2& range2,
                               Compare comp) {
    return std::lexicographical_compare
      (boost::begin(range1), boost::end(range1),
       boost::begin(range2), boost::end(range2),
       comp);
  }
  
  // Permutations
  //============================================================================
  template<class BidirectionalRange>
  bool next_permutation(BidirectionalRange& range) {
    return std::next_permutation(boost::begin(range), boost::end(range));
  }
  
  template<class BidirectionalRange, class Compare>
  bool next_permutation(BidirectionalRange& range, Compare comp) {
    return std::next_permutation(boost::begin(range), boost::end(range), comp);
  }
  
  template<class BidirectionalRange>
  bool prev_permutation(BidirectionalRange& range) {
    return std::prev_permutation(boost::begin(range), boost::end(range));
  }

  template<class BidirectionalRange, class Compare>
  bool prev_permutation(BidirectionalRange& range, Compare comp) {
    return std::prev_permutation(boost::begin(range), boost::end(range));
  }

  /*
  // Special versions of min & max
  //============================================================================
  //! Returns the minimum element of a non-empty collection.
  //! @require R::value_type is comparable
  template <typename R>
  typename boost::range_value<R>::type min(const R& values) {
    //concept_assert((InputRange<R>));
    assert(!boost::empty(values));
    return *min_element(values);
  }

  //! Returns the minimum element of a collection, with the given default value
  template <typename R>
  typename R::value_type min(const R& values, typename R::value_type init) {
    return accumulate(values, init, minimum<typename R::value_type>());
  }

  //! Returns the maximum element of a non-empty collection.
  //! @require R::value_type is comparable
  template <typename R>
  typename boost::range_value<R>::type max(const R& values) {
    //concept_assert((InputRange<R>));
    assert(!boost::empty(values));
    return *max_element(values);
  }

  //! Returns the maximum element of a collection, with the given default value
  template <typename R>
  typename R::value_type max(const R& values, typename R::value_type init) {
    return accumulate(values, init, maximum<typename R::value_type>());
  }

  //! Stores the indices of the true elements
  template <typename R, typename OutIt>
  OutIt find_indices(const R& values, OutIt out) {
    typedef typename R::size_type size_type;
    concept_assert((OutputIterator<OutIt, size_type>));

    size_type index = 0;
    foreach_auto(value, values) {
      if (value) (*out++) = index;
      ++index;
    }

    return out;
  }

  template <typename Container, typename Range>
  void append(Container& c, const Range& values) {
    c.insert(c.end(), boost::begin(values), boost::end(values));
  }

  //! Concatenates a sequence of vectors
  template <typename R>
  typename R::value_type vector_concat(const R& vectors) {
    typedef typename R::value_type vector;
    typedef typename vector::size_type size_type;
    // concept_assert((ublas::VectorConcept<vector>));

    // compute the size of the resulting vector
    size_type n = 0;
    foreach(const vector& v, vectors) n += v.size();
    vector result(n);

    // assign the vectors to the right indices
    n = 0;
    foreach(const vector& v, vectors) {
      subrange(result, n, n+v.size()) = v;
      n += v.size();
    }
    return result;
  }
  */
 
  //! @} // group range_algorithm

}

#include <prl/macros_undef.hpp>

#endif // PRL_ALGORITHM_HPP


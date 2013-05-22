#ifndef SILL_RANGE_CONCEPTS_HPP
#define SILL_RANGE_CONCEPTS_HPP

#include <boost/range/concepts.hpp>
#include <boost/range/iterator.hpp>
#include <boost/range/value_type.hpp>
#include <boost/type_traits/is_same.hpp>

#include <sill/global.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! \addtogroup range_concepts
  //! @{

  // Traversal
  //============================================================================
  template <typename R>
  struct SinglePassRange : boost::SinglePassRangeConcept<R> { };

  template <typename R>
  struct ForwardRange : boost::ForwardRangeConcept<R> { };

  template <typename R>
  struct BidirectionalRange : boost::BidirectionalRangeConcept<R> { };

  template <typename R>
  struct RandomAccessRange : boost::RandomAccessRangeConcept<R> { };

  // Access
  //============================================================================
  template <typename R>
  struct ReadableRange : SinglePassRange<R> {
    typedef typename boost::range_iterator<R>::type iterator;
    concept_assert((boost_concepts::ReadableIteratorConcept<iterator>));
  };

  template <typename R>
  struct WritableRange : SinglePassRange<R> {
    typedef typename boost::range_iterator<R>::type iterator;
    concept_assert((boost_concepts::WritableIteratorConcept<iterator>));
  };

  // Combinations
  //============================================================================
  template <typename R>
  struct InputRange : ReadableRange<R> { };

  template <typename R>
  struct ReadableForwardRange : ReadableRange<R>, ForwardRange<R> { };
  
  template <typename R, typename T>
  struct InputRangeConvertible : InputRange<R> {
    typedef typename boost::range_iterator<R>::type iterator;
    typedef typename std::iterator_traits<iterator>::reference reference;
    concept_assert((boost::Convertible<reference, T>));
  };

  template <typename R, typename T>
  struct ReadableForwardRangeConvertible : ReadableForwardRange<R> {
    typedef typename boost::range_iterator<R>::type iterator;
    typedef typename std::iterator_traits<iterator>::reference reference;
    concept_assert((boost::Convertible<reference, T>));
  };

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

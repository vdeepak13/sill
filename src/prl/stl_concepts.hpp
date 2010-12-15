#ifndef PRL_STL_CONCEPTS
#define PRL_STL_CONCEPTS

#include <boost/concept_check.hpp>

namespace prl {
  using boost::Integer;
  using boost::SignedInteger;
  using boost::UnsignedInteger;
  using boost::DefaultConstructible;
  using boost::Assignable;
  using boost::CopyConstructible;
  using boost::SGIAssignable;
  using boost::Convertible;
  using boost::EqualityComparable;
  using boost::LessThanComparable;
  using boost::Comparable;
  
  using boost::Generator;
  using boost::UnaryFunction;
  using boost::BinaryFunction;
  using boost::UnaryPredicate;
  using boost::BinaryPredicate;
  using boost::Const_BinaryPredicate;
  using boost::AdaptableGenerator;
  using boost::AdaptableUnaryFunction;
  using boost::AdaptableBinaryFunction;
  using boost::AdaptablePredicate;
  using boost::AdaptableBinaryPredicate;

  using boost::EqualityComparable;
  using boost::InputIterator;
  using boost::OutputIterator;
  using boost::ForwardIterator;
  using boost::Mutable_ForwardIterator;
  using boost::BidirectionalIterator;
  using boost::Mutable_BidirectionalIterator;
  using boost::RandomAccessIterator;
  using boost::Mutable_RandomAccessIterator;
                                            
  using boost::Container;
  using boost::Mutable_Container;
  using boost::ForwardContainer;
  using boost::Mutable_ForwardContainer;
  using boost::ReversibleContainer;
  using boost::Mutable_ReversibleContainer;
  using boost::RandomAccessContainer;
  using boost::Mutable_RandomAccessContainer;
  using boost::Sequence;  // Mutable_ForwardContainer + insert/erase
  using boost::FrontInsertionSequence;
  using boost::AssociativeContainer;
  using boost::UniqueAssociativeContainer;
  using boost::MultipleAssociativeContainer;
  using boost::SimpleAssociativeContainer;
  using boost::PairAssociativeContainer;
  using boost::SortedAssociativeContainer;

  //! A unique pair associative container, i.e., a map
  template <typename C>
  struct UniquePairAssociativeContainer :
    boost::UniqueAssociativeContainer<C>,
    boost::PairAssociativeContainer<C> { };

}  // namespace prl

#endif

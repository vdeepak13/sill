
#ifndef SILL_TRANSFORM_OUTPUT_ITERATOR_HPP
#define SILL_TRANSFORM_OUTPUT_ITERATOR_HPP

#include <iterator>

#include <sill/global.hpp>
#include <sill/stl_concepts.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * An output iterator that transforms the values that are assigned
   * to it and then assigns the result to another output iterator.
   *
   * @tparam OutIt a type that models the OutputIterator concept
   * @tparam F a type that models the UnaryFunction concept
   *
   * \ingroup iterator
   */
  template <typename OutIt, typename F>
  class transform_output_iterator
    : public std::iterator<std::output_iterator_tag, void, void, void, void> {
    concept_assert((OutputIterator<OutIt, typename F::result_type>));

  private:
    //! The underlying output iterator to which transformed values are assigned
    OutIt out;

    //! An unary function used to transform the values
    F f;

  public:
    //! Standard constructor
    transform_output_iterator(OutIt out, F f) : out(out), f(f) { }

    //! Assignment
    template <typename T>
    transform_output_iterator& operator=(const T& a) {
      *out = f(a);
      return *this; 
    }

    //! Simply returns *this (the iterator has no notion of a position)
    transform_output_iterator& operator*() {
      return *this;
    }

    //! Advances the output in place
    transform_output_iterator& operator++() {
      ++out;
      return *this;
    }

    //! Advances the output
    transform_output_iterator operator++(int) {
      transform_output_iterator it(*this);
      ++(*this);
      return it;
    }

    //! Returns the underlying output iterator
    OutIt iterator() const {
      return out; 
    }
  };

  //! A convenience function for constructing transform output iterators.
  //! \relates transform_output_iterator
  template <typename OutIt, typename F>
  transform_output_iterator<OutIt, F> transformed_output(OutIt out, F f) {
    return transform_output_iterator<OutIt, F>(out, f);
  }

} // namespace sill

#include <sill/macros_def.hpp>

#endif

#ifndef SILL_FACTOR_OPERATIONS_HPP
#define SILL_FACTOR_OPERATIONS_HPP

#include <sill/factor/combine_iterator.hpp>
#include <sill/factor/concepts.hpp>
#include <sill/factor/traits.hpp>
#include <sill/functional/inplace.hpp>
#include <sill/stl_concepts.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! \addtogroup factor_operations
  //! @{

  // Free functions that implement collapse-out style operations
  // ===========================================================================

  //! Returns the sum (integral) of a factor over a subset of variables
  template <typename F>
  typename F::collapse_type 
  sum(const F& f, const typename F::domain_type& eliminate) {
    // if the compilation fails here, F does not support marginalization
    return f.marginal(set_difference(f.arguments(), eliminate));
  }

  //! Returns the maximum of a factor over a subset of variabless
  template <typename F>
  typename F::collapse_type
  max(const F& f, const typename F::domain_type& eliminate) {
    // if the compilation fails here, F does not support maximization
    return f.maximum(set_difference(f.arguments(), eliminate));
  }

  //! Returns the minimum of a factor over a subset of variables
  template <typename F>
  typename F::collapse_type
  min(const F& f, const typename F::domain_type& eliminate) {
    // if the compilation fails here, F does not support minimization
    return f.minimum(set_difference(f.arguments(), eliminate));
  }

  // Functions on collections of factors
  // ===========================================================================

  /**
   * Combines a collection of factors.
   *
   * @param factors
   *        A collection that models the Boost SinglePassRange
   *        concept. The elements of the collection must
   *        satisfy the Factor concept.
   * @tparam Op inplace operation applied to the sequence
   */
  template <typename Op, typename Range>
  typename Range::value_type
  combine_all(const Range& factors, const typename Range::value_type& init) {
    concept_assert((InputRange<Range>));
    typedef typename Range::value_type factor_type;
    Op op;
    return sill::copy(factors, combine_iterator<factor_type, Op>(init, op)).result();
  }

  //! Multiplies a collection of factors that support operator*=
  template <typename Range>
  typename Range::value_type
  prod_all(const Range& factors) {
    typedef typename Range::value_type factor_type;
    return combine_all<inplace_multiplies<factor_type> >(factors, factor_type(1));
  }

  //! Sums a collection of factors that support operator+=
  template <typename Range>
  typename Range::value_type
  sum_all(const Range& factors) {
    typedef typename Range::value_type factor_type;
    return combine_all<inplace_plus<factor_type> >(factors, factor_type(0));
  }

  //! Returns the union of arguments of a collection of factors
  template <typename Range>
  typename Range::value_type::domain_type 
  arguments(const Range& factors) {
    concept_assert((InputRange<Range>));
    typename Range::value_type::domain_type args;
    foreach(const typename Range::value_type& f, factors) {
      args.insert(f.arguments().begin(), f.arguments().end());
    }
    return args;
  }  

  //! @} group factor_operations

} // namespace sill

#include <sill/macros_undef.hpp>

#endif


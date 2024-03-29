#ifndef SILL_FACTOR_OPERATIONS_HPP
#define SILL_FACTOR_OPERATIONS_HPP

#include <sill/factor/util/commutative_semiring.hpp>
#include <sill/factor/concepts.hpp>
#include <sill/stl_concepts.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! \addtogroup factor_operations
  //! @{

  // Free functions that implement collapse-out style operations
  // ===========================================================================

  //! Returns the sum (integral) of a factor over a subset of variables
  template <typename F>
  F sum(const F& f, const typename F::domain_type& eliminate) {
    // if the compilation fails here, F does not support marginalization
    return f.marginal(set_difference(f.arguments(), eliminate));
  }

  //! Returns the maximum of a factor over a subset of variabless
  template <typename F>
  F max(const F& f, const typename F::domain_type& eliminate) {
    // if the compilation fails here, F does not support maximization
    return f.maximum(set_difference(f.arguments(), eliminate));
  }

  //! Returns the minimum of a factor over a subset of variables
  template <typename F>
  F min(const F& f, const typename F::domain_type& eliminate) {
    // if the compilation fails here, F does not support minimization
    return f.minimum(set_difference(f.arguments(), eliminate));
  }

  // Functions on collections of factors
  // ===========================================================================

  //! Multiplies a collection of factors that support operator*=
  template <typename Range>
  typename Range::value_type
  prod_all(const Range& factors) {
    typedef typename Range::value_type factor_type;
    factor_type result(1);
    foreach(const factor_type& f, factors) {
      result *= f;
    }
    return result;
  }

  //! Sums a collection of factors that support operator+=
  template <typename Range>
  typename Range::value_type
  sum_all(const Range& factors) {
    typedef typename Range::value_type factor_type;
    factor_type result(0);
    foreach(const factor_type& f, factors) {
      result += f;
    }
    return result;
  }

  //! Combines a collection of factors using the specified commutative semiring
  template <typename Range>
  typename Range::value_type
  combine_all(const Range& factors,
              const commutative_semiring<typename Range::value_type>& csr) {
    concept_assert((InputRange<Range>));
    typedef typename Range::value_type factor_type;
    factor_type result = csr.combine_init();
    foreach(const factor_type& f, factors) {
      csr.combine_in(result, f);
    }
    return result;
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


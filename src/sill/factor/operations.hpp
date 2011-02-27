#ifndef SILL_FACTOR_OPERATORS_HPP
#define SILL_FACTOR_OPERATORS_HPP

#include <boost/utility/enable_if.hpp>

#include <sill/base/stl_util.hpp>
#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/combine_iterator.hpp>
#include <sill/factor/concepts.hpp>
#include <sill/factor/constant_factor.hpp>
//#include <sill/factor/gaussian_crf_factor.hpp>
#include <sill/factor/moment_gaussian.hpp>
//#include <sill/factor/table_crf_factor.hpp>
#include <sill/factor/traits.hpp>
#include <sill/math/gdl_enum.hpp>
#include <sill/range/algorithm.hpp>
#include <sill/stl_concepts.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Forward declarations
  class canonical_gaussian;
  class constant_factor;
//  class gaussian_crf_factor;
  class moment_gaussian;
//  class table_crf_factor;

  //! \addtogroup factor_operations
  //! @{

  // Standard free functions
  // ===========================================================================

  //! The default implementation of a combine operation for two identical
  //! factor types. Simply copies the factor and calls combine_in.
  template <typename F>
  typename combine_result<F, F>::type
  combine(F f1, const F& f2, op_type op) {
    concept_assert((Factor<F>));
    return f1.combine_in(f2, op);
  }

  /*
  //! The combine operation for two distinct factor types.
  template <typename F, typename G>
  typename combine_result<F, G>::type
  combine(const F& f, const G& g, op_type op);

  //! Specialization for constant_factor and table_crf_factor.
  template <>
  combine_result<constant_factor, table_crf_factor>::type
  combine<constant_factor, table_crf_factor>
  (const constant_factor& f, const table_crf_factor& g, op_type op);

  //! Specialization for constant_factor and gaussian_crf_factor.
  template <>
  combine_result<constant_factor, gaussian_crf_factor>::type
  combine<constant_factor, gaussian_crf_factor>
  (const constant_factor& f, const gaussian_crf_factor& g, op_type op);
  */

  /**
   * Combines a collection of factors.
   *
   * @param factors
   *        A collection that models the Boost SinglePassRange
   *        concept. The elements of the collection must
   *        satisfy the Factor concept.
   * @param op_type
   *        the combination operation.
   */
  template <typename Range>
  typename Range::value_type
  combine(const Range& factors, op_type op) {
    concept_assert((InputRange<Range>));
    typedef typename Range::value_type factor_type;
    return sill::copy(factors, combine_iterator<factor_type>(op)).result();
  }

  //! A free function version of a collapse operation
  template <typename F>
  typename F::collapse_type
  collapse(const F& f, op_type op, const typename F::domain_type& retain) {
    return f.aggregate(op, retain);
  }

  //! Collapses a factor expression over variables other than vars
  template <typename F>
  typename F::collapse_type
  collapse_out(const F& f, const typename F::domain_type& eliminate,
               op_type op) {
    return f.collapse(op, set_difference(f.arguments(),  eliminate));
  }

  //! Returns the sum (integral) of a factor over a subset of variables
  template <typename F>
  typename F::collapse_type 
  sum(const F& f, const typename F::domain_type& eliminate) {
    // if the compilation fails here, F does not support marginalization
    return collapse_out(f, eliminate, sum_op, 0.0);
  }

  //! Returns the maximum of a factor over a subset of variabless
  template <typename F>
  typename F::collapse_type
  max(const F& f, const typename F::domain_type& eliminate) {
    // if the compilation fails here, F does not support maximization
    return collapse_out(f, eliminate, max_op, 
                        -std::numeric_limits<double>::infinity());
  }

  //! Returns the minimum of a factor over a subset of variables
  template <typename F>
  typename F::collapse_type
  min(const F& f, const typename F::domain_type& eliminate) {
    // if the compilation fails here, F does not support minimization
    return collapse_out(f, eliminate, min_op, 
                        std::numeric_limits<double>::infinity());
  }

  //! Inverts a factor expression
  template <typename F>
  F invert(const F& f) {
    return F(1) / f;
  }

  //! Returns the union of arguments of a collection of factors
  template <typename FactorRange>
  typename FactorRange::value_type::domain_type 
  arguments(const FactorRange& factors) {
    concept_assert((InputRange<FactorRange>));
    typename FactorRange::value_type::domain_type args;
      
    foreach(const typename FactorRange::value_type& f, factors) {
      args = set_union(args, f.arguments());
    }
    return args;
  }

  // Standard operators
  // ===========================================================================

  //! Multiplies two factors
  template <typename F, typename G>
  typename combine_result<F,G>::type operator*(const F& f, const G& g) {
    // if the compilation fails here, the expression f * g is not valid
    return combine(f, g, product_op);
  }

  //! Divides two factors
  template <typename F, typename G>
  typename combine_result<F,G>::type operator/(const F& f, const G& g) {
    // if the compilation fails here, the expression f / g is not valid
    return combine(f, g, divides_op);
  }

  //! Adds two factors
  template <typename F, typename G>
  typename combine_result<F,G>::type operator+(const F& f, const G& g) {
    // if the compilation fails here, the expression f + g is not valid
    return combine(f, g, sum_op);
  }

  //! Subtracts one factor from another
  template <typename F, typename G>
  typename combine_result<F,G>::type operator-(const F& f, const G& g) {
    // if the compilation fails here, the expression f - g is not valid
    return combine(f, g, minus_op);
  }

  /**
   * Selective inclusion of templates.
   * @see http://www.boost.org/doc/libs/1_35_0/libs/utility/enable_if.html
   *
   * The following free functions are templatized based on two factor types.
   * If the function is not supported for a particular combination of factors,
   * e.g., f *= g where f is a table_factor and g is a Gaussian distribution,
   * the compilation will fail on the combine_in line.
   */
  using boost::enable_if;
  
  //! Multiplies in one factor to another
  template <typename F, typename G>
  typename enable_if<is_factor<F>, F&>::type operator*=(F& f, const G& g) {
    // if the compilation fails here, the expression f *= g is not valid
    return f.combine_in(g, product_op);
  }

  //! Divides in one factor to another
  template <typename F, typename G>
  typename enable_if<is_factor<F>, F&>::type operator/=(F& f, const G& g) {
    // if the compilation fails here, the expression f /= g is not valid
    return f.combine_in(g, divides_op);
  }

  //! Adds in one factor to another
  template <typename F, typename G>
  typename enable_if<is_factor<F>, F&>::type operator+=(F& f, const G& g) {
    // if the compilation fails here, the expression f += g is not valid
    return f.combine_in(g, sum_op);
  }

  //! Subtracts a factor in place
  template <typename F, typename G>
  typename enable_if<is_factor<F>, F&>::type operator-=(F& f, const G& g) {
    // if the compilation fails here, the expression f -= g is not valid
    return f.combine_in(g, minus_op);
  }
 
  //! @} group factor_operations

} // namespace sill

#include <sill/macros_undef.hpp>

#endif


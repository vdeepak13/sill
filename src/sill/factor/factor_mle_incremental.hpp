#ifndef SILL_FACTOR_MLE_INCREMENTAL_HPP
#define SILL_FACTOR_MLE_INCREMENTAL_HPP

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A utility class that represents a maximum-likelihood estimator
   * of the factor distribution that is able to process data incrementally.
   * The class constructor takes a vector of factor arguments (or the vectors
   * of head and tail arguments), and provides functions that process each
   * (weighted) data point and return the final estimate.
   *
   * By itself, this template is not capable of anything; it must be
   * specialized for each factor type that can be estimated incrementally.
   * The specializations must follow the interface declared in this class,
   * but may provide additional functionality as needed.
   *
   * \tparam Factor a type that satisfies the LearnableFactor and
   *                IndexableFactor concepts
   */
  template <typename Factor>
  class factor_mle_incremental {
  public:
    BOOST_STATIC_ASSERT_MSG(
      sizeof(Factor) == 0,
      "Missing specialization of factor_mle_incremental for the given factor type"
    );

    //! The type representing a real value
    typedef typename Factor::real_type real_type;

    //! The type representing a vector of Factor's arguments
    typedef typename Factor::var_vector_type var_vector_type;

    //! The type representing an index, i.e., a value in the Factor's domain
    typedef typename Factor::index_type index_type;
    
    //! The parameters of the estimator (regularization etc.)
    struct param_type;

    /**
     * Constructs a maximum-likelihood estimator over the given arguments.
     */
    factor_mle_incremental(const var_vector_type& args,
                           const param_type& params = param_type());

    /**
     * Constructs a maximum-likelihood estimator over the given head and
     * tail arguments.
     */
    factor_mle_incremental(const var_vector_type& head,
                           const var_vector_type& tail,
                           const param_type& params = param_type());
    
    /**
     * Incorporates a single data point represented by an index and its weight.
     * If computing a conditional distribution, the indices must be passed in
     * the (tail, head) order, not (head, tail)!
     */
    void process(const index_type& index, real_type weight);

    /**
     * Returns the final estimate of the factor's distribution.
     * Must not be called more than once.
     */
    Factor estimate();

    /**
     * Returns the total weight of the data incorporated into an estimate.
     */
    real_type weight() const;

  }; // class factor_mle_incremental

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

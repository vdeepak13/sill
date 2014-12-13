#ifndef SILL_FACTOR_EVALUATOR_HPP
#define SILL_FACTOR_EVALUATOR_HPP

namespace sill {

  /**
   * A helper class that evaluates the factor for the factor-specific
   * argument type. By default, the calls to this class simply get
   * forwarded to the underlying factor (which is stored by reference),
   * but implementations can perform partial template specialization
   * to optimize this functionality.
   *
   * \see factor_evaluator<moment_gaussian>
   * \todo Should we allow assignment_type as an argument as well?
   */
  template <typename F>
  class factor_evaluator {
  public:
    typedef typename F::result_type     result_type;
    typedef typename F::real_type       real_type;
    typedef typename F::index_type      index_type;
    typedef typename F::var_vector_type var_vector_type;
    
    //! Constructs the evaluator, holding the passed factor by reference
    factor_evaluator(const F& factor)
      : factor(factor) { }

    //! Evaluates the factor for the given argument
    result_type operator()(const index_type& arg) const {
      return factor(arg);
    }

    //! Returns the ordering of variables in the underlying factor
    const var_vector_type& arg_vector() const {
      return factor.arg_vector();
    }

  private:
    const F& factor;
  }; // class factor_evaluator

} // namespace sill

#endif

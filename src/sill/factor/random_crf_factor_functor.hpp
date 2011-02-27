#ifndef SILL_RANDOM_CRF_FACTOR_FUNCTOR_HPP
#define SILL_RANDOM_CRF_FACTOR_FUNCTOR_HPP

#include <sill/macros_def.hpp>

namespace sill {

  //! \addtogroup factor_random
  //! @{

  /**
   * Interface for functors for generating random crf_factor factors.
   * Such functors can then be plugged into methods for generating random
   * models (decomposable and crf_model).
   *
   * @see create_random_crf
   */
  template <typename F>
  struct random_crf_factor_functor {

    //! Type of Y in the conditional P(Y|X) this functor can generate.
    typedef OutputVariableType output_variable_type;

    //! Type of X in the conditional P(Y|X) this functor can generate.
    typedef InputVariableType input_variable_type;

    //! Create and return a random crf_factor.
    

  }; // struct random_crf_factor_functor

  //! @} group factor_random

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_RANDOM_CRF_FACTOR_FUNCTOR_HPP

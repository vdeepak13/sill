#ifndef SILL_NORMALIZATION_ERROR_HPP
#define SILL_NORMALIZATION_ERROR_HPP

#include <stdexcept>

namespace sill {

  /**
   * An exception thrown when a normalization operation cannot be performed on
   * model.
   *
   * @see decomposable::flow_functor, decomposable::flow_functor_mpa,
   *      decomposable::flow_functor_sampling, decomposable::operator*=,
   *      decomposable::replace_factors
   *
   * \ingroup model_exceptions
   */
  class normalization_error : public std::runtime_error {
  public:
    explicit normalization_error(const std::string& what)
      : std::runtime_error(what) { }
  };

}

#endif

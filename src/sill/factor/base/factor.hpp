#ifndef SILL_FACTOR_HPP
#define SILL_FACTOR_HPP

#include <sill/global.hpp>

namespace sill {

  /**
   * The base class of all factors. This class does not perform any 
   * functionality, but declares a virtual destructor, so that factors
   * can be allocated on the heap and cast if needed.
   *
   * For the functions provided by all factors, see the Factor concept.
   *
   * \ingroup factor_types
   */
  class factor {
  protected:
    //! The default constructor with no arguments
    factor() { }

  public:
    virtual ~factor() { }

  }; // class factor

} // namespace sill

#endif

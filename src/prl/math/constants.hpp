#ifndef PRL_MATH_CONSTANTS_HPP
#define PRL_MATH_CONSTANTS_HPP

#include <limits>
#include <boost/math/constants/constants.hpp>

namespace prl {

  //! \addtogroup math_constants
  //! @{

  // Constants for generic types
  //============================================================================
  //! Returns the infinity for the double floating point value
  inline double inf() {
    return std::numeric_limits<double>::infinity();
  }

  //! Returns the quiet NaN for the double floating point value
  inline double nan() {
    return std::numeric_limits<double>::quiet_NaN();
  }

  //! Returns pi for the double floating point type
  inline double pi() { 
    return boost::math::constants::pi<double>();
  }

  //! Returns root_pi for the double floating point type
  inline double root_pi() { 
    return boost::math::constants::root_pi<double>();
  }

  //! Returns root_half_pi for the double floating point type
  inline double root_half_pi() { 
    return boost::math::constants::root_half_pi<double>();
  }

  //! Returns root_two_pi for the double floating point type
  inline double root_two_pi() { 
    return boost::math::constants::root_two_pi<double>();
  }

  //! Returns root_ln_four for the double floating point type
  inline double root_ln_four() { 
    return boost::math::constants::root_ln_four<double>();
  }

  //! Returns e for the double floating point type
  inline double e() { 
    return boost::math::constants::e<double>();
  }

  //! Returns half for the double floating point type
  inline double half() { 
    return boost::math::constants::half<double>();
  }

  //! Returns eueler for the double floating point type
  inline double euler() { 
    return boost::math::constants::euler<double>();
  }

  //! Returns root_two for the double floating point type
  inline double root_two() { 
    return boost::math::constants::root_two<double>();
  }

  //! Returns ln_two for the double floating point type
  inline double ln_two() { 
    return boost::math::constants::ln_two<double>();
  }

  //! Returns ln_ln_two for the double floating point type
  inline double ln_ln_two() { 
    return boost::math::constants::ln_ln_two<double>();
  }

  //! Returns third for the double floating point type
  inline double third() { 
    return boost::math::constants::third<double>();
  }

  //! Returns two_thirds for the double floating point type
  inline double two_thirds() { 
    return boost::math::constants::twothirds<double>();
  }

  //! Returns pi_minus_three for the double floating point type
  inline double pi_minus_three() { 
    return boost::math::constants::pi_minus_three<double>();
  }

  //! Returns four_minus_pi for the double floating point type
  inline double four_minus_pi() { 
    return boost::math::constants::four_minus_pi<double>();
  }

  //! Returns pow23_four_minus_pi for the double floating point type
  inline double pow23_four_minus_pi() { 
    return boost::math::constants::pow23_four_minus_pi<double>();
  }

  //! Returns exp_minus_half for the double floating point type
  inline double exp_minus_half() { 
    return boost::math::constants::exp_minus_half<double>();
  }

  // @}

} // namespace prl

#endif

#ifndef SILL_FACTOR_TRAITS_HPP
#define SILL_FACTOR_TRAITS_HPP

#include <boost/type_traits.hpp>

namespace sill {

  //! \addtogroup factor_traits
  //! @{

  template <typename F>
  struct has_plus : public boost::false_type { };

  template <typename F>
  struct has_plus_assign : public boost::false_type { };

  template <typename F>
  struct has_minus : public boost::false_type { };

  template <typename F>
  struct has_minus_assign : public boost::false_type { };

  template <typename F>
  struct has_negate : public boost::false_type { };

  template <typename F>
  struct has_multiplies : public boost::false_type { };

  template <typename F>
  struct has_multiplies_assign : public boost::false_type { };

  template <typename F>
  struct has_divides : public boost::false_type { };

  template <typename F>
  struct has_divides_assign : public boost::false_type { };

  template <typename F>
  struct has_max : public boost::false_type { };

  template <typename F>
  struct has_max_assign : public boost::false_type { };

  template <typename F>
  struct has_min : public boost::false_type { };

  template <typename F>
  struct has_min_assign : public boost::false_type { };

  template <typename F>
  struct has_bit_and : public boost::false_type { };

  template <typename F>
  struct has_bit_and_assign : public boost::false_type { };

  template <typename F>
  struct has_bit_or : public boost::false_type { };

  template <typename F>
  struct has_bit_or_assign : public boost::false_type { };

  template <typename F>
  struct has_marginal : public boost::false_type { };

  template <typename F>
  struct has_maximum : public boost::false_type { };

  template <typename F>
  struct has_minimum : public boost::false_type { };

  template <typename F>
  struct has_logical_and : public boost::false_type { };

  template <typename F>
  struct has_logical_or : public boost::false_type { };

  template <typename F>
  struct has_arg_max : public boost::false_type { };

  template <typename F>
  struct has_arg_min : public boost::false_type { };

  //! @}

} // namespace sill

#endif

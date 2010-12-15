#ifndef PRL_FACTOR_TRAITS_HPP
#define PRL_FACTOR_TRAITS_HPP

#include <boost/type_traits/is_base_of.hpp>

// Factor traits
namespace prl {

  //! \addtogroup factor_operations
  //! @{

  /**
   * A class that specifies the result of a binary combine operation.
   * The factor clases need to specialize this template to provide
   * a nested type typedef.  By providing the template specialization,
   * the binary operators *, /, + are automatically implemented for
   * all supported cases.
   *
   * @see constant_factor, table_factor
   */
  template <typename F, typename G>
  struct combine_result { };

  // forward declaration
  class factor;

  /**
   * A class that specifies whether the template argument is a factor.
   * This class is used to automatically define the *=, /=, and += operators.
   * By default, all descendants of the factor class are marked as factors.
   */
  template <typename F>
  struct is_factor : boost::is_base_of<factor, F> { };

  //! @} group factor_types

} // namespace prl

#endif

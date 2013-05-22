add_subdirectory(linear_algebra)

add_prefix(PRL_LINEAR_ALGEBRA_SOURCES "linear_algebra/")

set(PRL_MATH_SOURCES
  ${PRL_LINEAR_ALGEBRA_SOURCES}
  linear_algebra
  operations
  statistics
  function/logistic_discrete
  multinomial_distribution
  PARENT_SCOPE
)

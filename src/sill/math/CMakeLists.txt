add_subdirectory(linear_algebra)

add_prefix(SILL_LINEAR_ALGEBRA_SOURCES "linear_algebra/")

set(SILL_MATH_SOURCES
  ${SILL_LINEAR_ALGEBRA_SOURCES}
  linear_algebra
  operations
  statistics
  function/logistic_discrete
  multinomial_distribution
  PARENT_SCOPE
)

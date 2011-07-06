add_subdirectory(sparse_linear_algebra)

add_prefix(PRL_SPARSE_LINEAR_ALGEBRA_SOURCES "sparse_linear_algebra/")

set(PRL_MATH_SOURCES
  ${PRL_SPARSE_LINEAR_ALGEBRA_SOURCES}
#  matrix
#  vector
  linear_algebra
  statistics
  function/logistic_discrete
  multinomial_distribution
  PARENT_SCOPE
)

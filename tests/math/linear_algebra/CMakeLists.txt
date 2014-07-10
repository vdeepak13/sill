
add_executable(sparse_linear_algebra_test sparse_linear_algebra_test.cpp)
add_executable(sparse_vector sparse_vector.cpp)

if(!CYGWIN)
  add_executable(vector_matrix_ops vector_matrix_ops.cpp)
endif(!CYGWIN)

link_libraries(itpp)

subdirs(function sparse_linear_algebra)

#add_executable(gdl gdl.cpp)
add_executable(multinomial_distribution multinomial_distribution.cpp)
#add_executable(quaternion quaternion.cpp)
add_executable(test_matrix test_matrix.cpp)
add_executable(test_vector test_vector.cpp)

if (SUITESPARSE_FOUND)
  add_executable(suitesparse_test suitesparse_test.cpp)
endif (SUITESPARSE_FOUND)

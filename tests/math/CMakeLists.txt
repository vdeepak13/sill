subdirs(linear_algebra)

#add_executable(functions functions.cpp)
add_executable(logarithmic logarithmic.cpp)
add_executable(multinomial_distribution multinomial_distribution.cpp)

if (SUITESPARSE_FOUND)
  add_executable(suitesparse_test suitesparse_test.cpp)
endif (SUITESPARSE_FOUND)

#add_test(functions functions)
add_test(logarithmic logarithmic)
add_test(multinomial_distribution multinomial_distribution)

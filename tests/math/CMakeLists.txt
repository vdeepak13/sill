subdirs(eigen likelihood random)

add_executable(logarithmic logarithmic.cpp)

if (SUITESPARSE_FOUND)
  add_executable(suitesparse_test suitesparse_test.cpp)
endif (SUITESPARSE_FOUND)

add_test(logarithmic logarithmic)

project(tests)

link_libraries(sill ${Boost_LIBRARIES})

# Disable unused variable warnings in tests
if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wno-unused-variable")
endif(CMAKE_COMPILER_IS_GNUCXX)
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -frepo")

subdirs(
  base
  datastructure
  factor
  graph
  inference
  iterator
  learning
  model
  math
  mpi
  optimization
  parsers
  synthetic_data
  threading
  serialization
  )

add_executable(copy_ptr copy_ptr.cpp)
add_executable(tr1_test tr1_test.cpp)
add_executable(tr1_unordered_map tr1_unordered_map.cpp)


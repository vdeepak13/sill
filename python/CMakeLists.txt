find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
link_libraries(${PYTHON_LIBRARIES} sill ${Boost_LIBRARIES})

add_library(sillpy SHARED 
  sill.cpp
  base.cpp
  factor.cpp
  graph.cpp
  model.cpp
  converters.cpp)
#add_library(factor SHARED factor.cpp converters.cpp)

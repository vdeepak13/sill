## Main SILL project file The cmake documentation is available online
## or in your installation cmake e.g. "C:\Program Files\CMake
## 2.6\doc\cmake-2.6"

project(SILL)

cmake_minimum_required(VERSION 2.8.3)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

enable_testing()

# add a prefix to each element of a list
function(add_prefix list_name prefix)
  # create empty list - necessary?
  SET(${list_name}_TMP)
 
  # prefix and suffix elements
  foreach(l ${${list_name}})
    list(APPEND ${list_name}_TMP ${prefix}${l})
  endforeach()
 
  # replace list by tmp list
  SET(${list_name} "${${list_name}_TMP}" PARENT_SCOPE)
endfunction(add_prefix)

# Specify the subdirectories for this root directory This will be
# used for the class branch, not the lab branch: subdirs(src examples
# EXCLUDE_FROM_ALL tests timings projects) (Remove EXCLUDE_FROM_ALL
# for lab branch.)
subdirs(src examples tests timings tools) # python)

# This would make the if statements easier to read
# set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)
# set(CMAKE_VERBOSE_MAKEFILE ON) -- supposedly useful for EclipseCDT
# http://www.vtk.org/Wiki/CMake:Eclipse_UNIX_Tutorial#
# Automatic_Discovery_of_Include_directories_.28Optional.2C_but_handy.29

# add random environment variable definitions
# deal with Boost time shtuff: we want nanoseconds!
# add_definitions(-DBOOST_DATE_TIME_POSIX_TIME_STD_CONFIG)
# add_definitions(-DBOOST_ALL_DYN_LINK)

if (MSVC)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS) # disable checked_iterator warnings
  add_definitions(-D_CRT_SECURE_NO_WARNINGS) # disable fopen warnings
  add_definitions(-D_SECURE_SCL=0)  # necesary for release builds of MSVC
endif(MSVC)

#MESSAGE(STATUS "CMAKE_SYSTEM: ${CMAKE_SYSTEM}")

if (${CMAKE_SYSTEM} MATCHES Linux)
  link_directories(/usr/local/lib64)  # Joseph: issues with bigbros
  set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS ON)
endif (${CMAKE_SYSTEM} MATCHES Linux)

# Remove tracing by default
# add_definitions(-DTRACE_OFF)

## Here we use our cmake tools
set(CMAKE_MODULE_PATH ${SILL_SOURCE_DIR}/cmake)

#set(BUILD_SHARED_LIBS ON)

# set(FindBoost_DIR ${SILL_SOURCE_DIR}/cmake)
# find_package(FindBoost)
if(MSVC)
  set(Boost_USE_STATIC_LIBS OFF)
  set(Boost_USE_MULTITHREADED ON)
  add_definitions(-DBOOST_ALL_NO_LIB=1)
else(MSVC)
  set(Boost_USE_STATIC_LIBS ON)
  set(Boost_USE_MULTITHREADED ON)
endif(MSVC)

find_package(Boost 1.40 COMPONENTS 
              program_options 
              filesystem
              iostreams  # deprecate together with MPI
              date_time
              thread
              # python
              unit_test_framework
	      system REQUIRED)

# Sparse linear algebra library
find_package(SuiteSparse)

# Armadillo
find_package(Armadillo REQUIRED)

# set include path for this and all subdirs
include_directories(
  ${SILL_SOURCE_DIR}/src
  ${Boost_INCLUDE_DIR}
  ${ARMADILLO_INCLUDE_DIRS}
  )

link_directories(
  ${Boost_LIBRARY_DIRS}
  )

link_libraries(
  # ${Boost_LIBRARIES}
  # Boost libraries these are now included in tests/ etc. to ensure proper
  # ordering libraries in static linking.
  ${ARMADILLO_LIBRARIES}
  #  mpichcxx
  #  mpich
  )

if(!MINGW)
link_libraries(bz2)
endif(!MINGW)

if (SUITESPARSE_FOUND)
  include_directories(${SUITESPARSE_INCLUDE_DIRS})
#  link_directories(${SUITESPARSE_LIBRARY_DIRS})
  link_libraries(${SUITESPARSE_LIBRARIES})
endif (SUITESPARSE_FOUND)

if (${CMAKE_SYSTEM} MATCHES Darwin)
#  link_libraries(blas)
  set(CMAKE_EXE_LINKER_FLAGS "-framework Accelerate")
endif (${CMAKE_SYSTEM} MATCHES Darwin)

# Add posix threads on non-windows sytems
if(NOT ${CMAKE_SYSTEM} MATCHES Windows)
  link_libraries(pthread)   
endif(NOT ${CMAKE_SYSTEM} MATCHES Windows)

# Flags for the debug and release mode
if(CMAKE_COMPILER_IS_GNUCXX)
  # Note: -fkeep-inline-functions significantly slows down the compilation
  set(CMAKE_CXX_FLAGS_DEBUG "-g -ggdb -O0 -Wall -Winit-self") #-fno-implicit-templates")
  # set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -ggdb")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
else()
  set(CMAKE_CXX_FLAGS "-std=c++11")
endif(CMAKE_COMPILER_IS_GNUCXX)

# uncomment the following line to get CMake variables to print to screen
# include(CMakePrintSystemInformation)

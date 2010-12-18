## Main SILL project file The cmake documentation is available online
## or in your installation cmake e.g. "C:\Program Files\CMake
## 2.6\doc\cmake-2.6"

project(SILL)

cmake_minimum_required(VERSION 2.6)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)


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
subdirs(src examples tests timings)

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
              iostreams
              date_time
              thread
	      system REQUIRED)

# MPI Support for parallel algorithms 
find_package(MPICH2)
#find_package(MPI)

# I don't know how to do this automatically
set(MPI_CXX true CACHE BOOL
    "Set to true if the MPI implementation provides a C++ interface.")

# Math library
find_package(IT++)

# CUDA libraries (required only for GPU code)
#   The SDK and the Thrust library must be installed.
#   If CUDA libraries are not found, ignore the following commands:
#     cuda_add_executable
set(CUDA_NVCC_FLAGS_Debug "-g -D_DEBUG -G")
mark_as_advanced(CUDA_NVCC_FLAGS_Debug)
find_package(CUDA)
# Check for the CUDA SDK.
if(CUDA_FOUND)
  if(CUDA_SDK_ROOT_DIR)
    message(STATUS "CUDA SDK found: " ${CUDA_SDK_ROOT_DIR})
  else(CUDA_SDK_ROOT_DIR)
    message(STATUS "CUDA_SDK_ROOT_DIR not found! CUDA will be set to not found. Install the SDK, or define the directory explicitly if it exists.")
    set(CUDA_FOUND "")
  endif(CUDA_SDK_ROOT_DIR)
endif(CUDA_FOUND)
# Check for CUDA Thrust.
if(CUDA_FOUND)
  find_package(CUDAThrust QUIET)
  if(CUDATHRUST_FOUND)
    message(STATUS "CUDA Thrust found: " ${CUDATHRUST_INCLUDE_DIR})
  else(CUDATHRUST_FOUND)
    message(STATUS "CUDA Thrust not found.  Some executables will not be compiled.")
  endif(CUDATHRUST_FOUND)
endif(CUDA_FOUND)
if(CUDA_FOUND)
  message(STATUS "CUDA Found: " ${CUDA_VERSION})
else(CUDA_FOUND)
  message(STATUS "CUDA Not Found! GPU executables will not be compiled.")
  macro(CUDA_ADD_EXECUTABLE cuda_target)
  endmacro(CUDA_ADD_EXECUTABLE cuda_target)
endif(CUDA_FOUND)

# OpenCV (optional)
# If OpenCV is not found, projects/qsgm/vision/ compiles to throw an error.
find_package(OpenCV)

# set include path for this and all subdirs
include_directories(
  ${SILL_SOURCE_DIR}/src
  ${ITPP_INCLUDE_DIR}
  ${Boost_INCLUDE_DIR}
  )

link_directories(
  ${Boost_LIBRARY_DIRS}
  ${ITPP_LINK_DIR}
  )

link_libraries(
  # ${Boost_LIBRARIES}
  # Boost libraries these are now included in tests/ etc. to ensure proper
  # ordering libraries in static linking.
  ${ITPP_LIBRARIES}
  itpp # this can be changed to itpp_debug to turn on bound checking etc.
  #  mpichcxx
  #  mpich
  bz2
  )

# If we find MPI link against it
if(MPI_FOUND)
  include_directories(${MPI_INCLUDE_PATH})
  link_directories(${MPI_INCLUDE_PATH})
  link_libraries(${MPI_LIBRARY} ${MPI_EXTRA_LIBRARY})
  # Stano: not sure if these do anything but the link flags are
  # non-empty for openmpi
  set(CMAKE_CXX_FLAGS "${MPI_COMPILE_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${MPI_LINK_FLAGS}") 
endif(MPI_FOUND)

if (${CMAKE_SYSTEM} MATCHES Darwin)
  link_libraries(blas)
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
endif(CMAKE_COMPILER_IS_GNUCXX)

# uncomment the following line to get CMake variables to print to screen
# include(CMakePrintSystemInformation)

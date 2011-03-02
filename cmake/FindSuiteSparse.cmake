# Defines
#
#  SUITESPARSE_FOUND - the system has SuiteSparse
#  SUITESPARSE_INCLUDE_DIRS - the include directories
#  SUITESPARSE_LIBRARY_DIRS - the link directories
#  SUITESPARSE_LIBRARIES - the libraries needed to use SuiteSparse
#
# Note:
#  A system wide installed CSPARSE is mapped to CXSPARSE on nearly all systems
#  (tested on FreeBSD, Debian GNU/Linux, Gentoo)
#
#=============================================================================
# This CMake file was adapted from the FindCSPARSE.cmake file provided here:
#  http://www-user.tu-chemnitz.de/~komart/software/cmake/FindCSPARSE.cmake
# This was the original copyright info accompanying the file:
#-----------------------------------------------------------------------------
# Copyright 2010, Martin Koehler
# http://www-user.tu-chemnitz.de/~komart/
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#-----------------------------------------------------------------------------
# Redistribution and use of this modified version is allowed according
# to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#=============================================================================

if(SUITESPARSE_INCLUDE_DIRS AND SUITESPARSE_LIBRARIES)
   # in cache already
   set(SuiteSparse_FIND_QUIETLY TRUE)
endif(SUITESPARSE_INCLUDE_DIRS AND SUITESPARSE_LIBRARIES)

if(MSVC)
  set(SUITESPARSE_FOUND FALSE) # TO DO
else (MSVC)
  # Find CXSparse
  #--------------------------------------------------------
  find_path(CXSPARSE_INCLUDE_DIR cs.h
    /usr/include
    /usr/local/include
    /usr/local/include/suitesparse
    /usr/include/suitesparse
    )
  find_library(CXSPARSE_LIBRARY NAMES ${CXSPARSE_NAMES} cxsparse libcxsparse
    PATHS /usr/lib /usr/local/lib /usr/local/lib/suitesparse
    )
  if (CXSPARSE_LIBRARY)
    set(CXSPARSE_LIBRARIES ${CXSPARSE_LIBRARY} )
  endif (CXSPARSE_LIBRARY)
  if (CXSPARSE_INCLUDE_DIR AND CXSPARSE_LIBRARY)
    set(CXSPARSE_FOUND TRUE)
  else (CXSPARSE_INCLUDE_DIR AND CXSPARSE_LIBRARY)
    set(CXSPARSE_FOUND FALSE)
  endif (CXSPARSE_INCLUDE_DIR AND CXSPARSE_LIBRARY)
  mark_as_advanced(CXSPARSE_INCLUDE_DIR CXSPARSE_LIBRARY)

  # Find next SuiteSparse package... (TO DO)
  #--------------------------------------------------------

  # Combine everything under the SUITESPARSE name.
  #--------------------------------------------------------
  set (SUITESPARSE_INCLUDE_DIRS ${CXSPARSE_INCLUDE_DIR})
  set (SUITESPARSE_LIBRARIES ${CXSPARSE_LIBRARY})
  if (CXSPARSE_FOUND)
    set (SUITESPARSE_FOUND TRUE)
  else (CXSPARSE_FOUND)
    set (SUITESPARSE_FOUND FALSE)
  endif (CXSPARSE_FOUND)
endif (MSVC)

if (SUITESPARSE_FOUND)
  if (NOT SuiteSparse_FIND_QUIETLY)
    message(STATUS "Found SuiteSparse: ${SUITESPARSE_LIBRARIES}")
  endif (NOT SuiteSparse_FIND_QUIETLY)
else (SUITESPARSE_FOUND)
  if (SuiteSparse_FIND_REQUIRED)
    message(SEND_ERROR "Could NOT find SuiteSparse")
  else (SuiteSparse_FIND_REQUIRED)
    if (NOT SuiteSparse_FIND_QUIETLY)
      message(STATUS "Could not find SuiteSparse")
    endif (NOT SuiteSparse_FIND_QUIETLY)
  endif (SuiteSparse_FIND_REQUIRED)
endif (SUITESPARSE_FOUND)

mark_as_advanced(SUITESPARSE_INCLUDE_DIRS SUITESPARSE_LIBRARIES)

# This file defines a variable HAVE_Chord
# Set this variable to true (using ccmake), in order to build functionality
# that requires Chord. This requires both Chord and sfslite-0.8 to be 
# installed in the /usr/local (default) and the GMP libraries in standard
# search path.
#
# Additional variables: 
# Chord_INCLUDE_DIR contains the necessary include files
# Chord_LIBRARIES contains the necessary libraries

set(HAVE_Chord false CACHE BOOL 
    "Set to true if the reference implementation of Chord is installed in /usr/local.")

if (HAVE_Chord)
  set(Chord_INCLUDE_DIRS
    /usr/local/include/chord-0.1 
    /usr/local/include/sfslite-0.8)
  set(Chord_LIBRARIES
    /usr/local/lib/chord-0.1/libutil.a
    /usr/local/lib/chord-0.1/libsvc.a
#    /usr/local/lib/sfslite-0.8/libtame.a
#    /usr/local/lib/sfslite-0.8/libsfsmisc.a
#    /usr/local/lib/sfslite-0.8/libsvc.a
    /usr/local/lib/sfslite-0.8/libsfscrypt.a
    /usr/local/lib/sfslite-0.8/libarpc.a
    /usr/local/lib/sfslite-0.8/libasync.a
    gmp
    )
endif (HAVE_Chord)


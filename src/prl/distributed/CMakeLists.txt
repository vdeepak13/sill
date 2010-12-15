set(PRL_DISTRIBUTED_SOURCES
  list_random_host.cpp
  address_functions.cpp
  )

if (HAVE_Chord)
  include_directories(${Chord_INCLUDE_DIRS})
  set(PRL_DISTRIBUTED_SOURCES 
    ${PRL_DISTRIBUTED_SOURCES} 
    chord_lookup.cpp
    chord_successor.cpp
    chord_predecessor.cpp
    chord_successor_list.cpp
    chord_random_host.cpp
    chordID_wrapper.cpp
    chordID_range.cpp
    sfslite_io.cpp
    )
endif (HAVE_Chord)

add_library(prl-distributed ${PRL_DISTRIBUTED_SOURCES})

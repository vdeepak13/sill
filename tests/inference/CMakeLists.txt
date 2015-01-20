subdirs(
  exact
  loopy
  mpi
  parallel
  sampling
  variational
)

#add_executable(markov_logic markov_logic.cpp)

find_package(TCMALLOC)
if(TCMALLOC_FOUND) 
  target_link_libraries(markov_logic tcmalloc)
  target_link_libraries(protein_inference tcmalloc)
endif()

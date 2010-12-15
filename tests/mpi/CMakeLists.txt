
subdirs(
  experiments
  )


if (MPI_FOUND)

# Example code from mpich for computing pi
add_executable(cpi cpi.c)

# # Tests for the PRL mpi tools
# add_executable(mpi_tests mpi_tests.cpp)
# add_executable(mpilogtest mpilogtest.cpp)
# add_executable(mpi_state_manager_tests mpi_state_manager_tests.cpp)
# # add_executable(mpi_state_adapter_tests mpi_state_adapter_tests.cpp)

# add_executable(mpi_sync_denoise mpi_sync_denoise.cpp)
# target_link_libraries(mpi_sync_denoise jpeg tcmalloc)

# add_executable(mpi_state_manager_denoise mpi_state_manager_denoise.cpp)
# target_link_libraries(mpi_state_manager_denoise jpeg tcmalloc)

  add_executable(mpi_send_receive mpi_send_receive)
  add_executable(mpi_broadcast mpi_broadcast)
  add_executable(mpi_tree_aggregate mpi_tree_aggregate)
  add_executable(mpi_line_aggregate mpi_line_aggregate)
endif (MPI_FOUND)

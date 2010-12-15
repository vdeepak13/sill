
if (MPI_FOUND AND MPI_CXX)
  add_library(prl-mpi
    mpi_wrapper mpilogger mpi_single_tag_handler mpi_consensus)
else (MPI_FOUND AND MPI_CXX)
  add_library(prl-mpi dummy.cpp)
endif (MPI_FOUND AND MPI_CXX)

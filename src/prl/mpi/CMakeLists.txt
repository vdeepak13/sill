if (MPI_FOUND AND MPI_CXX)
  set(PRL_MPI_SOURCES
    mpi_wrapper mpilogger mpi_single_tag_handler mpi_consensus PARENT_SCOPE)
else (MPI_FOUND AND MPI_CXX)
  set(PRL_MPI_SOURCES PARENT_SCOPE)
endif (MPI_FOUND AND MPI_CXX)

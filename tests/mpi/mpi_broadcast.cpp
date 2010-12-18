#include <cstdio>

#include <sill/mpi/mpi_bcast.hpp>

int main(int argc, char** argv) {
  using namespace sill;

  int numprocs, myid;  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  std::string value;

  if (myid == 0)
    value = "Hello";

  mpi_bcast(0, value);
  printf("%d: %s\n", myid, value.c_str());

  MPI_Finalize();
  return 0;
}

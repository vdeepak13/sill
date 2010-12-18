#include <cstdio>

#include <sill/mpi/mpi_tree_aggregate.hpp>
#include <sill/functional/add.hpp>

// aggregates data along a spanning tree
int main(int argc, char** argv) {
  using namespace sill;
  using namespace std;
  
  int myid;  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  int total;
  mpi_tree_aggregate(add<int>(), myid, total);
  printf("Process %d: total = %d\n", myid, total);

  MPI_Finalize();
}

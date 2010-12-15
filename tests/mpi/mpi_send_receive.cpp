#include <iostream>

#include <prl/mpi/mpi_send_recv.hpp>


int main(int argc, char** argv) {
  using namespace prl;

  int numprocs, myid;  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  std::cout << "Process " << myid << std::endl;
  
  if (myid == 0) {
    mpi_send(1, 0, std::string("Hello"));
    std::string msg;
    mpi_recv(1, 1, msg);
    std::cout << msg << "!" << std::endl;
  } else {
    std::string msg;
    mpi_recv(0, 0, msg);
    std::cout << msg << ", ";
    std::cout.flush();
    mpi_send(0, 1, std::string("world"));
  }

  MPI_Finalize();
  return 1;
}

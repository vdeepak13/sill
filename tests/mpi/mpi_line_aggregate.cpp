#include <cstdio>

#include <boost/random/mersenne_twister.hpp>

#include <sill/stl_io.hpp>
#include <sill/mpi/mpi_line_aggregate.hpp>
#include <sill/functional/add.hpp>
#include <sill/parallel/pthread_tools.hpp>
#include <sill/math/free_functions.hpp>

boost::mt19937 rng;

struct send_line : public sill::runnable {

  //std::vector<int> sources;
  size_t group_size;

  send_line(size_t group_size) : group_size(group_size) { }

  void run() {
    using namespace sill;
    int numprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    for(int i = 0; i < numprocs; i++) {
      mpi_send(i, 0, i == 0 ? -1 : i-1);
      mpi_send(i, 0, i == numprocs-1 ? -1 : i+1);
      mpi_send(i, 0, numprocs);
    }
  } 
//     std::vector<size_t> perm = randperm(numprocs, rng);
//     std::cerr << perm << std::endl;
//     for (size_t i = 0; i < (size_t)numprocs; i++) {
//       printf("%d: %d\n", int(i), int(perm[i]));
//       // the previous node
//       if (i % group_size == 0)
//         mpi_send(perm[i], 0, -1);
//       else
//         mpi_send(perm[i], 0, perm[i-1]);

//       // the next node
//       if ((i+1) % group_size == 0 || i == (size_t)numprocs-1)
//         mpi_send(perm[i], 0, -1);
//       else
//         mpi_send(perm[i], 0, perm[i+1]);
      
//       // the group size
//       if (i >= (numprocs/group_size)*group_size)
//         mpi_send(perm[i], 0, numprocs % group_size);
//       else
//         mpi_send(perm[i], 0, group_size);
//     }
//   }

};

// aggregates data along a spanning tree
int main(int argc, char** argv) {
  using namespace sill;
  using namespace std;
  
  int myid;  
  int numprocs;
  int avail_threads;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &avail_threads);
  assert(avail_threads == MPI_THREAD_MULTIPLE);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

  send_line sl(10);
  thread_group g;
  if (myid == 0) g.launch(&sl);

  int prev, next, count;
  mpi_recv(0, 0, prev);
  mpi_recv(0, 0, next);
  mpi_recv(0, 0, count);
  printf("Node %d: prev %d, next %d, count %d\n", myid, prev, next, count);

  if (myid == 0) g.join();
    
  int total;
  mpi_line_aggregate(prev, next, add<int>(), myid, total);
  printf("Process %d: total = %d\n", myid, total);

  MPI_Finalize();
}

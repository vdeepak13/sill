#ifndef PRL_MPI_TREE_AGGREGATE_HPP
#define PRL_MPI_TREE_AGGREGATE_HPP

#include <prl/mpi/mpi_send_recv.hpp>

#include <prl/functional/aggregate_op.hpp>

namespace prl {

  /**
   * Aggregates the values from all the processes and disseminates the value.
   */
  template <typename T>
  void mpi_tree_aggregate(const aggregate_op<T>& op,
                          const T& value, 
                          T& aggregate) {
    int numprocs, myid;  
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // receive the partial aggregates from the children
    T partial_aggregate = value;
    if (2*myid < numprocs && myid != 0) {
      T x;
      mpi_recv(2*myid, 0, x);
      op(x, partial_aggregate);
    }
    if (2*myid + 1 < numprocs) {
      T x;
      mpi_recv(2*myid + 1, 0, x);
      op(x, partial_aggregate);
    }
  
    // send the partial aggregates to the parent
    if (myid > 0) {
      mpi_send(myid / 2, 0, partial_aggregate);
    }

    // receive the aggregate from the parent
    if (myid > 0) 
      mpi_recv(myid / 2, 0, aggregate);
    else
      aggregate = partial_aggregate;
    
    // distribute out the result to children
    if (2*myid < numprocs && myid != 0) 
      mpi_send(2*myid, 0, aggregate);

    if (2*myid + 1 < numprocs)
      mpi_send(2*myid + 1, 0, aggregate);
  }
  
} // namespace prl

#endif

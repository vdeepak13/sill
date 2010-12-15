#ifndef PRL_MPI_LINE_AGGREGATE_HPP
#define PRL_MPI_LINE_AGGREGATE_HPP

#include <prl/mpi/mpi_send_recv.hpp>

#include <prl/functional/aggregate_op.hpp>

namespace prl {

  /**
   * Aggregates the values from all the processes and disseminates the value.
   */
  template <typename T>
  void mpi_line_aggregate(int prev,
                          int next,
                          const aggregate_op<T>& op,
                          const T& value, 
                          T& aggregate) {
    // receive the partial aggregates from the previous node
    T partial_aggregate = value;
    if (prev >= 0) {
      T x;
      mpi_recv(prev, 0, x);
      op(x, partial_aggregate);
    }
  
    // send the partial aggregates to the next node
    if (next >= 0) {
      mpi_send(next, 0, partial_aggregate);
    }

    // receive the result from the next node
    if (next >= 0) 
      mpi_recv(next, 0, aggregate);
    else
      aggregate = partial_aggregate;
    
    // send the result to the previous node
    if (prev >= 0)
      mpi_send(prev, 0, aggregate);

  }
  
} // namespace prl

#endif

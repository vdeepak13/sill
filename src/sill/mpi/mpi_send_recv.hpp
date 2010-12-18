#ifndef SILL_MPI_SEND_RECV_HPP
#define SILL_MPI_SEND_RECV_HPP

#include <mpi.h>

#include <string>
#include <sstream>
#include <limits>

#include <sill/serialization/serialize.hpp>

namespace sill {
  
  template <typename T>
  void mpi_send(int dest, int tag, const T& value) {
    std::ostringstream out(std::ios_base::binary);
    oarchive ar(out);
    ar << value;
    std::string str = out.str();
    
    // send the size
    assert(str.size() <= unsigned(std::numeric_limits<int>::max()));
    int size = str.size();
    MPI_Send(&size, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
    
    // send the content
    MPI_Send(const_cast<char*>(str.c_str()), size, MPI_CHAR, dest, tag,
             MPI_COMM_WORLD);
  }

  template <typename T>
  MPI_Status mpi_recv(int src, int tag, T& value) {

    // receive the size
    int size = 0;
    int source;
    MPI_Status status;
    MPI_Recv(&size, 1, MPI_INT, src, tag, MPI_COMM_WORLD, &status);
    assert(size>=0);
    source = status.MPI_SOURCE;
    //std::cout << size << std::endl;
    
    // receive the content 
    // note that we receive the content from the same source, not src
    // (in case src is MPI_ANY_SOURCE)
    std::string str(size, 0);
    MPI_Recv(const_cast<char*>(str.c_str()), size, MPI_CHAR, source, tag,
             MPI_COMM_WORLD, &status);
    assert(source == status.MPI_SOURCE);
    std::istringstream in(str, std::ios_base::binary);
    iarchive ar(in);
    ar >> value;
    
    return status;
  }

} // namespace sill

#endif

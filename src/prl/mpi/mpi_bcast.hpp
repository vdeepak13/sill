#ifndef PRL_MPI_BCAST_HPP
#define PRL_MPI_BCAST_HPP

#include <mpi.h>

#include <string>
#include <sstream>
#include <limits>

#include <prl/serialization/serialize.hpp>

namespace prl {

  template <typename T>
  void mpi_bcast(int root, T& value) {
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    if (root == myid) { // sender
      std::ostringstream out(std::ios_base::binary);
      oarchive ar(out);
      ar << value;
      std::string str = out.str();
    
      // broadcast the size
      assert(str.size() <= unsigned(std::numeric_limits<int>::max()));
      int size = str.size();
      MPI_Bcast(&size, 1, MPI_INT, root, MPI_COMM_WORLD);
      
      // broadcast the serialized content
      MPI_Bcast(const_cast<char*>(str.c_str()), size, MPI_CHAR, root,
                MPI_COMM_WORLD);
    } else { // receiver
      // receive the size
      int size = 0;
      MPI_Bcast(&size, 1, MPI_INT, root, MPI_COMM_WORLD);
      assert(size>=0);
    
      // receive the serialized content 
      std::string str(size, 0);
      MPI_Bcast(const_cast<char*>(str.c_str()), size, MPI_CHAR, root,
                MPI_COMM_WORLD);

      // deserialize the content
      std::istringstream in(str, std::ios_base::binary);
      iarchive ar(in);
      ar >> value;
    }
  }

} // namespace prl

#endif

#ifndef MPI_PROTOCOLS_HPP
#define MPI_PROTOCOLS_HPP
#include <climits>

namespace sill{
  namespace MPI_WRAPPER_PROTOCOL_FLAGS {   
    // IDs 0-7 are reserved for internal messaging
    const int NUM_PROTOCOL_BITS = 16;
    const int PROTOCOL_BITMASK = (1 << 16) - 1; 
    const int FLAG_UNCOUNTED = 1 << 16;
    const int FLAG_UNCOUNTED_BIT = 16;
  } // namespace MPI_Protocol

  namespace MPI_PROTOCOL {   
    // IDs 0-7 are reserved for internal messaging
    const int STOP = 0;
    const int CONSOLE_IO = 1;
    const int CONSENSUS = 2;
    const int INTERNAL_UPPER_ID = 7;
    const int GLOBAL_UPPER_ID = MPI_WRAPPER_PROTOCOL_FLAGS::PROTOCOL_BITMASK;
    // bits 17-31 are reserved for internal signalling
  } // namespace MPI_Protocol
  
}

#endif //  MPI_PROTOCOLS_HPP

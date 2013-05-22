#include <sstream>
#include <string>
#include <iostream>
#include <sill/mpi/mpi_wrapper.hpp>
#include <sill/mpi/mpi_protocols.hpp>
namespace sill {

class mpilogger {
private:
  std::stringstream buffer;
  bool registered;
  mpi_post_office *post_office;
  static mpilogger* instance;
public:
  // this is a singleton
  mpilogger();
  ~mpilogger();
  bool register_mpi(mpi_post_office &po);


	template <typename T>
	mpilogger& operator<<(const T t) {
		buffer << t;
    // we flush on a '\n' or if the buffer is too long
    if (buffer.str()[buffer.str().length() - 1] == '\n') {
			while(1) {
				std::string s;
				std::getline(buffer, s);
				if (s.length() == 0) break;
				post_office->send_message(0, MPI_PROTOCOL::CONSOLE_IO, s.length(), s.c_str());
			}
	  }
	  return *this;
  }
};

extern mpilogger mpilog;


};

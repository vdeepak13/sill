#include <sill/mpi/mpilogger.hpp>

namespace sill {
mpilogger mpilog;

mpilogger* mpilogger::instance = NULL;

class mpilogger_text_handler: public mpi_post_office::po_box_callback {
public:
  void recv_message(const mpi_post_office::message& msg) {
    std::cout << msg.body;
  }
  void terminate() { } 
};

bool mpilogger::register_mpi(mpi_post_office &po) {
  post_office = &po;
  po.register_handler(MPI_PROTOCOL::CONSOLE_IO, new mpilogger_text_handler());
  return true;
}

mpilogger::mpilogger() {
  if (instance !=NULL) {
    throw("multiple instances of mpilogger");
  }
  else {
    instance = this;
  }
}

mpilogger::~mpilogger() {
}

}

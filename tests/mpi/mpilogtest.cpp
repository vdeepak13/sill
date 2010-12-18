// This file tests mpi routines
#include <string>
#include <stdlib.h>
#include <iostream>

#include <sill/mpi/mpi_wrapper.hpp>
#include <sill/mpi/mpilogger.hpp>
using namespace sill;
using namespace std;


// class stop_handler : public mpi_post_office::po_box_callback {
//   mpi_post_office* m_po;
// public:
//   stop_handler(mpi_post_office* po) : m_po(po) { } 
//   void recv_message(const mpi_post_office::message& msg) {
//     cout << "received stop message at process  " << m_po->id() 
//          << " on machine " << m_po->name() << endl;
//     // Shut down the post office
//     m_po->stop();
//     cout << "invoked stop at process " << m_po->id() 
//          << " on machine " << m_po->name() << endl;
//   }
// };


int main(int argc,char ** argv) {
  cout << "Testing PRL mpi tools" << endl;

  mpi_post_office mpi;
  
  cout << "Id:    " << mpi.id() << endl
       << "Name:  " << mpi.name() << endl
       << "Count: " << mpi.num_processes() << endl;

  cout << "Registering handlers." << endl;
  mpilog.register_mpi(mpi);

  cout << "Launching threads" << endl;
  mpi.start();

  cout << "Testing if process 0" << endl;

  if(mpi.id() != 0) {
    sleep(1);
    mpilog << "Hello from  " << mpi.id() << '\n';
	}
	else {
    sleep(5); 
    mpi.stopAll();
  }
  cout << "Beginning wait on process: " << mpi.id() << endl;
  mpi.wait();
  cout << "Finished wait on process: " << mpi.id() << endl;
  return EXIT_SUCCESS;
}

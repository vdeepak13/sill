
// This file tests mpi routines

#include <string>
#include <stdlib.h>
#include <iostream>
#include <cstdlib>


#include <prl/mpi/mpi_wrapper.hpp>
#include <prl/parallel/pthread_tools.hpp>

using namespace std;
using namespace prl;



class text_handler : public mpi_post_office::po_box_callback {
public:
  void recv_message(const mpi_post_office::message& msg) {
    // Shut down the post office
    cout << "Texthandler: " << msg << endl;
  }
  void terminate() {  }
};



void test_mpi_po() {
  const size_t TEXT_ID = 20;

//   cout << "testing mpi PO" << endl;

  mpi_post_office mpi;
  cout << "Name:  " << mpi.name() << endl;

  cout << "Registering handlers." << endl;
  mpi.register_handler(TEXT_ID, new text_handler());


  mpi.start();

  if(mpi.id() == 0) {
    cout << mpi.name() << ": I am Process 0." << endl;
    cout << "Sending Hello" << endl;
    for(size_t i = 0; i < mpi.num_processes(); ++i ) {
      string name = "Hello from Process 0";
      mpi.send_message(i, TEXT_ID, name.size(), name.c_str());
    }
    cout << "Sending goodbye" << endl;
    for(size_t i = 0; i < mpi.num_processes(); ++i ) {
      string name = "Goodbye from Process 0";
      mpi.send_message(i, TEXT_ID, name.size(), name.c_str());
    }
    cout << "Sending Stop" << endl;
    // Stopping process 0
    mpi.stopAll();
    cout << "Finished all sending at process 0" << endl;
  } 
  mpi.wait();
}

class wait_thread : public thread {
  MPI::Request& req_;
public:
  wait_thread(MPI::Request& req) : 
    req_(req) {

  }
  void run() {
    std::cout << "Beginning to wait" << std::endl;
    
    req_.Wait();
    
    std::cout << "No longer waiting" << std::endl;
  }

};

void cancel_test() {
  MPI::Init_thread(MPI::THREAD_MULTIPLE);
  size_t body_size = 3;
  char* body[3];
  MPI::Request request = MPI::COMM_WORLD.Irecv(body,
                                               body_size,
                                               MPI::CHAR,
                                               MPI::ANY_SOURCE,
                                               MPI::ANY_TAG);
  

  wait_thread thr(request);
  thr.start();
  sleep(2);
  std::cout << "Invoking cancel" << std::endl;
  request.Cancel();
  std::cout << "Finished call to cancel" << std::endl;
  sleep(5);
  std::cout << "Finalizing" << std::endl;
  MPI::Finalize();
  std::cout << "DONE" << std::endl;
}


int main(int argc, char* argv[]) {
  cancel_test();
   
  return EXIT_SUCCESS;
}

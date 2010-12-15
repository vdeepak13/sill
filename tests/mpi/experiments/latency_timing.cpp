

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <string>

#include <boost/program_options.hpp>

// #include <boost/archive/text_oarchive.hpp>
// #include <boost/archive/text_iarchive.hpp>




#include <mpi.h>


#include <prl/parallel/pthread_tools.hpp>
#include <prl/parallel/timer.hpp>

using namespace std;
using namespace prl;




class shared_object {
  mutex mutex_;
  conditional conditional_;
  vector<size_t> data_;
  size_t user_count_;
  size_t turn_;
  size_t updates_;
  size_t max_updates_;
public:
  shared_object(size_t buffer_size, size_t maxupdates, size_t user_count) : 
    data_(buffer_size), user_count_(user_count), turn_(0),
    updates_(0), max_updates_(maxupdates) { }

  bool update(size_t id) {
    mutex_.lock();
//     while(turn_ != id && updates_ < max_updates_) {
//       conditional_.wait(mutex_);
//     }
    //     std::cout << updates_ << ", " << max_updates_ << std::endl;
    if(updates_ < max_updates_) {
      //      std::cout << id << ": " << "updating" << std::endl;
      data_[0] = data_[data_.size()-1];
      for(size_t i = 1; i < data_.size(); ++i) {
        data_[i] = (1 + data_[i-1]) * (1 + data_[i-1]);
      }
      updates_++;
    }
    bool finished = updates_ < max_updates_;
    turn_ = (turn_ + 1) % user_count_;
    //    conditional_.signal();
    //     std::cout << "User: " << id << " finished round" << std::endl;
    mutex_.unlock();
    return finished;
  }

  size_t sum() { 
    size_t ret = 0;
    for(size_t i = 0; i < data_.size(); ++i) {
      ret += data_[i];
    }
    return ret;
  }

};

class shared_memory_agent : public runnable {
  shared_object* obj_;
  size_t id_;
public:
  shared_memory_agent() {}
  void init(shared_object* obj, size_t id) {
    obj_ = obj;
    id_ = id;
  }
  void run() {
    for(bool alive = true; alive; ) {
      alive = obj_->update(id_);
    }
  }
};
   
void test_shared(size_t user_count, size_t updates, size_t buffer) {
  size_t max_updates = updates*user_count;
  shared_object shared_obj(buffer, max_updates, user_count);
  vector<shared_memory_agent> agents(user_count);
  thread_group threads;
  timer time;
  cout << "Running Experiment" << endl;
  time.start();
  for(size_t i = 0; i < agents.size(); ++i) {
    agents[i].init(&shared_obj, i);
    threads.launch(&(agents[i]));
  }
  threads.join();
  double running_time = time.current_time();
  cout << "Finished!" << endl;
  cout << "Sum: " << shared_obj.sum() << endl;
  cout << "Running Time: " << running_time << endl;
  cout << "Output: 1, " << user_count << ", " 
       << buffer << ", " 
       << updates << ", " 
       << running_time << endl;
} 



void test_mpi(size_t updates, size_t data_size) {
  // Start MPI
  MPI::Init();


  int id = MPI::COMM_WORLD.Get_rank();
  int user_count = MPI::COMM_WORLD.Get_size();
  int message_type = 0;

  int source = id-1;
  if(source < 0) source = user_count - 1;
  int dest = id+1;
  if(dest >= user_count) dest = 0;

  size_t* data = new size_t[data_size];
  for(size_t i = 0; i < data_size; ++i){
    data[i] = 0;
  }

  if(id == 0) cout << "Running mpi experiment" << endl;
  
  timer time; 
  if(id == (user_count-1)) time.start();
  for(size_t t = 0; t < updates; ++t) {
    if(!(id == 0 && t == 0)) {
      MPI::COMM_WORLD.Recv(data, data_size, MPI::UNSIGNED_LONG,
                           source, message_type);
    }
    data[0] = data[data_size-1];
    for(size_t i = 1; i < data_size; ++i) {
      //      std::cout << "(" << data[i-1] << ", ";
      data[i] = (1 + data[i-1]) * (1 + data[i-1]);
      //      std::cout <<  data[i] << "), ";
    }
    //    std::cout << endl;
    if( !(id == (user_count - 1) && t == (updates-1)) ) {
      MPI::COMM_WORLD.Send(data, data_size, MPI::UNSIGNED_LONG,
                           dest, message_type);
    }
  }

  if(id == (user_count-1)) {
    double running_time = time.current_time();
    size_t total = 0;
    for(size_t i = 0; i < data_size; ++i) {
      total += data[i];
    }
    std::cout << "Sum: " << total << std::endl;
    std::cout << "Running Time: " << running_time << std::endl;
    cout << "Output: 2, " 
         << user_count << ", " 
         << data_size << ", " 
         << updates << ", " 
         << running_time << endl;
    
  }

  delete [] data;

  MPI::Finalize();
}


int main(int argc, char* argv[]) {
  string type;
  size_t users;
  size_t updates;
  size_t buffer;

  namespace po = boost::program_options;
  po::options_description desc("Tests latency of access modalities");
  //  po::options_description desc;
  desc.add_options()
    ("help", "produce help message")
    ("type", po::value<string>(&type),  "Do [shared] or [mpi] memory timing")
    ("users", po::value<size_t>(&users), "number of threads")
    ("updates", po::value<size_t>(&updates), "number of updates per thread")
    ("buffer", po::value<size_t>(&buffer), "size of the buffer");
  
  // Specify the order of ops
  //  po::positional_options_description pos_opts; 
  po::variables_map vm;
  store(po::parse_command_line(argc, argv, desc), vm);
  notify(vm);

  if(vm.count("help") > 0 ||  vm.count("type") == 0 ||
     ((vm.count("users") == 0)  && (type == "shared")) || 
     vm.count("updates") == 0 || vm.count("buffer") == 0 ) {
    cout << "Usage: " << argv[0] << "[options]" << endl;
    cout << desc;
    return EXIT_FAILURE;
  }
  
  if(type == "shared") {
    cout << "Running shared experiment" << endl;
    test_shared(users, updates, buffer);
  } else if (type == "mpi") {
    test_mpi(updates, buffer);
  } else {
    cout << "Invalid type" << endl;
    cout << "Usage: " << argv[0] << "[options]" << endl;
    cout << desc;
    return EXIT_FAILURE;
  }

  

  return EXIT_SUCCESS;
}

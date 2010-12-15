#include <prl/parallel/pthread_tools.hpp>
#include <iostream>
#include <cassert>

using namespace prl;


template <typename LockType>
class lock_test_thread:public runnable {
  public:
    lock_test_thread(int *i, LockType *m) {
      i_ = i;
      m_ = m;
    }
    virtual void run() {
      m_->lock();
      (*i_)++;
      m_->unlock();
    }
    virtual ~lock_test_thread(){};
  private:
    int *i_;
    LockType *m_;
};

template <typename LockType>
void test_lock() {
  int i = 0;
  size_t numcpus = 0;
  LockType m;
  thread_group g;
  numcpus = thread::cpu_count();
  std::cout << numcpus << " CPUs detected" << std::endl;
  i = 0;
  for (size_t t = 0; t < numcpus; ++t) {
    g.launch(new lock_test_thread<LockType>(&i,&m), t);
  }
  g.join();
  assert(i==(int)numcpus);
  
	i = 0;
  for (size_t t = 0; t < 100; ++t) {
    g.launch(new lock_test_thread<LockType>(&i,&m));
  }
  g.join();
  assert(i==100);
}


int main(int argc,char** argv) {
  test_lock<mutex>();
  std::cout << "Mutex: ok\n";
  if (!SPINLOCK_SUPPORTED) {
    std::cout << "Spin lock not supported.\n";
  }
  else {
      std::cout << "Spin lock supported!\n";
      test_lock<spinlock>();
      std::cout << "Spinlock: ok\n";
  }
}

#include <prl/parallel/binned_mutable_queue.hpp>
#include <prl/parallel/pthread_tools.hpp>
#include <cassert>
#include <map>

using namespace prl;

void ShortCorrectnessTest(){
  binned_mutable_queue<int,int> q(2);
  q.push(10,10);
  q.push(5,5);
  q.push(11,11);
  q.push(4,4);
  q.push(6,6);
  q.push(5,5);  
  q.push(1,1);  
  q.push(15,15);
  
  assert(q.get(10)==10);
  assert(q.get(1)==1);
  assert(q.top_priority()==15);
  
  std::pair<int,int> t = q.pop();
  
  assert(t.first == 15 && t.second==15);
  
  q.remove(5);

  assert(q.get(5)<0);
  assert(q.get_unsafe(11)==11);

  q.promote(4,100);

  assert(q.top_priority()==100);
  
  q.pop();
  
  assert(q.top().first == 11);
  
  q.clear();
}


class synctest : public runnable {
  private:
    binned_mutable_queue<int,int> *q_;

  public:
    synctest(binned_mutable_queue<int,int> *q) {
      q_ = q;
    }

    virtual void run() {
      srand(time(NULL));
      for (size_t i=0;i < 300000;++i) {
        int r = rand() % 100;
        q_->push(r,r);
        if (i % 10==5) {
          std::pair<int,int> t = q_->pop();
          if (t.first != t.second && t.second >=0) {
            std::cout << t.first << " " << t.second << std::endl;
            assert(t.first == t.second);
          }
        }
      }
    }
    virtual ~synctest(){};
};


// this just test for races and deadlocks
void SynchronizationTest1(){
  binned_mutable_queue<int,int> q(10);
  thread_group g;
	for (size_t t = 0; t < 10; ++t) {
    g.launch(new synctest(&q));
  }
  g.join();
}


class synctest2:public runnable {
  public:
    synctest2(binned_mutable_queue<int,int> *q) {
      q_ = q;
    }
    virtual void run() {
      srand(time(NULL));
      for (size_t i=0;i < 1000000;++i) {
        int r = rand() % 100000;
        int s = rand() % 100000;
        q_->update_unsafe(r+1,s);
      }
    }
    virtual ~synctest2(){};
  private:
    binned_mutable_queue<int,int> *q_;
};


// this just test for races and deadlocks
void SynchronizationTest2(){
  binned_mutable_queue<int,int> q(10);
  for (size_t i=0;i < 100000;++i) {
    q.push(i+1,i+1);
  }
  thread_group g;
	for (size_t t = 0; t < 2; ++t) {
    g.launch(new synctest2(&q));
  }
  g.join();
}


int main(int argc,char ** argv) {
  std::cout<<"Short Correctness Test: ";
  std::cout.flush();
  ShortCorrectnessTest();
  std::cout<<"Ok\n";
  
  std::cout<<"Safe Sync Test 1: ";
  std::cout.flush();
  SynchronizationTest1();
  std::cout<<"Ok\n";
  
  std::cout<<"UnSafe Sync Test 2: ";
  std::cout.flush();
  SynchronizationTest2();
  std::cout<<"Ok\n";
}

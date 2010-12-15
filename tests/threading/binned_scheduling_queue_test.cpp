#include <prl/parallel/binned_scheduling_queue.hpp>
#include <prl/parallel/pthread_tools.hpp>
#include <cassert>
#include <map>
#include <vector>


using namespace prl;

void ShortCorrectnessTest(){
  binned_scheduling_queue<int> q(2);
  std::vector<std::pair<int,double> > v;
  v.push_back(std::pair<int,double>(10,10.0));
  v.push_back(std::pair<int,double>(5,5.0));
  v.push_back(std::pair<int,double>(11,11.0));
  v.push_back(std::pair<int,double>(4,4.0));
  v.push_back(std::pair<int,double>(6,6.0));
  v.push_back(std::pair<int,double>(5,5.0));
  v.push_back(std::pair<int,double>(1,1.0));
  v.push_back(std::pair<int,double>(15,15.0));
  q.init(v);
  assert(q.get(10)==10);
  assert(q.get(1)==1);
  assert(q.top_priority()==15);
  
  std::pair<int,int> t = q.top();
  
  assert(t.first == 15 && t.second==15);
  
  q.deschedule(5);

  assert(q.isscheduled(5)==false);
  
  q.promote(5,10.0);
  assert(q.get(5)==10.0);

  q.promote(4,100);
  assert(q.get(4)==100.0);
  
  q.schedule(5);
  q.schedule(4);
  
  assert(q.top().first == 4);
  assert(q.get(5)==10.0);
  assert(q.get(4)==100.0);
  
  q.deschedule_top();
  
  assert(q.top().first == 15);
  
  q.clear();
}


class synctest:public runnable {
  public:
    synctest(binned_scheduling_queue<int> *q) {
      q_ = q;
    }
    virtual void run() {
      srand(time(NULL));
      for (size_t i=0;i < 1000000;++i) {
        q_->deschedule_top();
				int r = rand() % 100000;
        int s = rand() % 100000;
        q_->promote(r+1,s+1);
				r = rand() % 100000;
        s = rand() % 100000;
        q_->update(r+1,s+1);
        
        r = rand() % 100000;
        q_->schedule(r + 1);
      }
    }
    virtual ~synctest(){};
  private:
    binned_scheduling_queue<int> *q_;
};


// this just test for races and deadlocks
void SynchronizationTest(){
  binned_scheduling_queue<int> q(10);
  std::vector<std::pair<int,double> > v;
  for (size_t i=0;i < 100000;++i) {
    v.push_back(std::pair<int,double>(i+1,i+1));
  }
	q.init(v);
	
  thread_group g;
	for (size_t t = 0; t < 2; ++t) {
    g.launch(new synctest(&q));
  }
  g.join();
}


int main(int argc,char ** argv) {
  std::cout<<"Short Correctness Test: ";
  std::cout.flush();
  ShortCorrectnessTest();
  std::cout<<"Ok\n";
 
  std::cout<<"Sync Test: ";
  std::cout.flush();
  SynchronizationTest();
  std::cout<<"Ok\n";
  
}

#ifndef SILL_WORKER_GROUP_HPP
#define SILL_WORKER_GROUP_HPP

#include <sill/global.hpp>
#include <sill/parallel/blocking_queue.hpp>
#include <sill/parallel/pthread_tools.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  template <typename Job, typename T = void_>
  class worker_group {
  public:
    worker_group(size_t nworkers,
                 boost::function<T(Job)> processor,
                 boost::function<T(T,T)> aggregator = NULL,
                 T init = T())
      : processor_(processor),
        aggregator_(aggregator),
        workers_(nworkers, worker(this, init)) {
      for (size_t i = 0; i < nworkers; ++i) {
        threads_.launch(&workers_[i]);
      }
    }

    void enqueue(const Job& item) { 
      queue_.enqueue(item);
    }

    template <typename Range>
    void enqueue_all(const Range& range) {
      foreach(Job item, range) {
        queue_.enqueue(item);
      }
    }

    void join() {
      queue_.wait_until_empty();
      queue_.stop_blocking();
      threads_.join();
    }

    T aggregate_result() const {
      T result = workers_[0].aggregate;
      for (size_t i = 1; i < workers_.size(); ++i) {
        result = aggregator_(result, workers_[i].aggregate);
      }
      return result;
    }

  private:
    struct worker : public runnable {
      worker_group* group;
      T aggregate;

      worker(worker_group* group, T init)
        : group(group), aggregate(init) { }

      void run() {
        std::pair<Job, bool> job;
        while ((job = group->queue_.dequeue()).second) {
          T result = group->processor_(job.first);
          aggregate = group->aggregator_(aggregate, result);
        }
      }
    };

    blocking_queue<Job> queue_;
    boost::function<T(Job)> processor_;
    boost::function<T(T,T)> aggregator_;
    std::vector<worker> workers_;
    thread_group threads_;

  }; // class worker_group
  
} // namespace sill

#include <sill/macros_undef.hpp>

#endif

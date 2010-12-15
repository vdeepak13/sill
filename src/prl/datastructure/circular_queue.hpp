#ifndef CIRCULAR_QUEUE
#define CIRCULAR_QUEUE

#include <cstddef>
#include <vector>
#include <list>


#include <prl/parallel/pthread_tools.hpp>

using namespace prl;
/**
  Provides a circular ordering over a fixed set of elements
*/
template<typename T>
class circular_queue {
private:
  typename std::vector<T>::const_iterator m_iterator;
  size_t m_cycles;
  const std::vector<T>& m_queue; 
  bool m_thread_safe;
  spinlock mut;
 
public:
  circular_queue(const std::vector<T>& queue, bool thread_safe = true) : 
    m_iterator(queue.begin()), m_cycles(0), m_queue(queue), 
    m_thread_safe(thread_safe) { }

  const T& next() {
    if(m_thread_safe) mut.lock();    
    const T& v = *m_iterator++;
    if(m_iterator == m_queue.end()) {
      m_cycles++;
      m_iterator = m_queue.begin();
    }
    if(m_thread_safe) mut.unlock();
    return v;
  }

  size_t size() { return m_queue.size(); } 
  size_t cycles() { return m_cycles; }
};

#endif

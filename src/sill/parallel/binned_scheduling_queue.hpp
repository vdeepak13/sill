#ifndef BINNED_SCHEDULING_QUEUE_HPP
#define BINNED_SCHEDULING_QUEUE_HPP

#include <iostream>
#include <map>
#include <cassert>
#include <cfloat>

#include <sill/parallel/binned_mutable_queue.hpp>
// #include <boost/unordered_map.hpp>


// This should come last
#include <sill/macros_def.hpp>

namespace sill {

  /**
   * \file binned_scheduling_queue.hpp
   * 
   * \class binned_scheduling_queue This is a queue used to keep a
   * scheduling order over arbitrary objects. The elements in the
   * queue are fixed at construction and cannot be changed at runtime.
   *
   * Elements can be "activated" whereby they will fall out of the
   * schedule. Activated elements can then be "deactivated" and it
   * will return into the schedule.
   *
   * To increase performance, the queue is randomly broken up into N
   * priority queues. So to get the top element, (N-1) comparisons and
   * N mutex lock are necesary. However, promotion and updates can be
   * performed with only one lock.
   *
   * This therefore sacrifices performance of the top() and pop()
   * functions, for increased parallelism of element updates.
   *
   * constraints: 
   *    - Priorities are doubles
   *    - priorities are strictly > 0 
   */
  template <typename T>
  class binned_scheduling_queue {

  private:
    binned_mutable_queue<T,double> m_q;
 

  public:
    binned_scheduling_queue(size_t bin_count = 5):m_q(bin_count) {}

    //! Enqueues a new item in the queue.
    void push(T item, double priority) {
      m_q.push(item,priority);
    }

    /** 
     * Initializes the queue the bin_count and inserts the elements
     * into the queue
     */
    void init(size_t bin_count, 
              const std::vector<std::pair<T,double> > &elements) {
      m_q.init(bin_count);
      init(elements);
    }
  
    //! inserts the elements into the queue
    void init(const std::vector<std::pair<T,double> > &elements) {
      for (size_t i = 0; i < elements.size(); ++i) {
        assert(elements[i].second >= DBL_MIN);
        m_q.push(elements[i].first,elements[i].second);
      }
    }
    
    /** inserts the elements between the iterators begin and end 
    	into the queue. The iterators must iterate over std::pair<T,double> 
    */
    template<class Iterator>
    void init(Iterator begin, Iterator end) {
      Iterator i = begin;
      while (i!=end){
        assert(i->second >= DBL_MIN);
        m_q.push(i->first,i->second);
        ++i;
      }
    }
        
    /** inserts the elements between the iterators begin and end,
    	giving all of them the initial priority p.
    	into the queue. The iterators must iterate over T.
    */
    template<class Iterator>
    void init(Iterator begin, Iterator end, double p) {
	     assert(p >= DBL_MIN);
      Iterator i = begin;
      while (i!=end){
        m_q.push(*i,p);
        ++i;
      }
    }
    
    //! Returns the number of elements in the heap.
    size_t size() const { return m_q.size(); }

    //! Returns true iff the queue is empty.
    bool empty() const { return m_q.empty(); }

    //! Returns true if the queue contains the given value
    bool contains(const T item) const { 
      return m_q.contains_unsafe(item);
    }

    //! Clears all the values (equivalent to stl clear)
    void clear() {
      m_q.clear();
    }

    /** 
     * Gets the top item, returning the item and its original priority
     * The item will now be descheduled and its priority will be set
     * to minimum priority.
     *
     * If there are no scheduled items, the priority value in the
     * returned pair will be < 0 and the queue will not be changed
     */
    inline std::pair<T,double> deschedule_top() {
      // if queue is empty return
      bool noresult = true;
      std::pair<T, double> res;
      while(noresult) {
        if (empty()) {
          res.second = -1;
          return res;
        }
			
        // I must lock this early otherwise there will be a race
        // condition in between getting the top element and removing
        // it from the inverse_map
			
        int max_index = -1;
        double max_priority = 0;
        for(size_t i = 0; i < m_q.m_bins.size(); ++i) {
          if(m_q.m_bins[i].empty() == false && 
             m_q.m_bins[i].top().second > max_priority) {
            max_index = i; max_priority = m_q.m_bins[i].top().second;
          }
        }
        if (max_priority < 0) {
          res.second = -1;
          return res;
        }
        if (max_index >= 0) {
          m_q.m_mutexs[max_index].lock();
          if (m_q.m_bins[max_index].empty() == false) {
            res = m_q.m_bins[max_index].top();
            // deschedule it. push it to negative priority
            m_q.m_bins[max_index].update(res.first, -DBL_MIN);
            noresult = false;
            m_q.m_mutexs[max_index].unlock();
          }
          else {
            m_q.m_mutexs[max_index].unlock();
          }
        }
      }
      return res;
    }

    //! Deschedules the item 'id'
    inline void deschedule(T id) {
      m_q.update_unsafe(id, -DBL_MIN);
    }

    //! Returns true if item 'id' is scheduled
    inline bool isscheduled(T id) {
      return m_q.get_unsafe(id) > 0;
    }

    inline const std::pair<T,double> top() const{
      return m_q.top_unsafe();
    }


    /** 
     * Returns the priority of the top scheduled item
     *
     * If there are no scheduled items, 
     * the return value will be < 0
     */
    inline double top_priority() {
      return m_q.top_priority_unsafe();
    }

    /** 
     * Returns the priority of item 'id' Assertion failure if id is
     *  not found in the queue.
     */
    inline double get(const T id) {
      assert(contains(id));
      double p = m_q.get_unsafe(id);
      if (p<0) return -p;
      else return p;
    }

    /** 
     * Returns the priority of item 'id' Assertion failure if id is
     *  not found in the queue.
     */
    inline double operator[](const T id) {
      return get(id);
    }

    /** 
     * This schedules item 'id' if it is current descheduled Assertion
     *  failure if id is not found in the queue.
     */
    inline void schedule(T id) {
      assert(contains(id));
      int bin_index = m_q.inverse_map_unsafe(id);
      m_q.m_mutexs[bin_index].lock();
      double p = m_q.m_bins[bin_index].get(id);
      if (p<0) {
        m_q.m_bins[bin_index].update(id,-p);
      }
      m_q.m_mutexs[bin_index].unlock();
    }

    /** 
     * Increases the priority of 'item' if its current priority is
     *  below new_prority. If the element is descheduled it stays
     *  descheduled.
     */
    inline void promote(const T item, double new_priority) {  
      if (new_priority < DBL_MIN) new_priority = DBL_MIN;
  	
      int bin_index = m_q.inverse_map_unsafe(item);
      m_q.m_mutexs[bin_index].lock();
      double p = m_q.m_bins[bin_index].get(item);
      if (p<0){
        p = std::min(p,-new_priority);
      }
      else{
        p = std::max(p,new_priority);
      }
      m_q.m_bins[bin_index].update(item, p);
      m_q.m_mutexs[bin_index].unlock();
    }

    /** 
     * Increases the priority of 'item' if its current priority is
     *  below new_prority. If the element is descheduled it stays
     *  descheduled.
     */
    inline void increase_priority(const T item, double delta) {  
      int bin_index = m_q.inverse_map_unsafe(item);
      m_q.m_mutexs[bin_index].lock();
      double p = m_q.m_bins[bin_index].get(item);
      if (p<0){
        p = p - delta;
      }
      else{
        p = p + delta;
      }
      m_q.m_bins[bin_index].update(item, p);
      m_q.m_mutexs[bin_index].unlock();
    }
    
    /** 
     * Changes the priority of 'item' to new_prority.  If the element
     * is descheduled it stays descheduled.
     */
    inline void update(const T item, double new_priority) {
      if (new_priority < DBL_MIN) new_priority = DBL_MIN;
  	
      int bin_index = m_q.inverse_map_unsafe(item);
      m_q.m_mutexs[bin_index].lock();
      double p = m_q.m_bins[bin_index].get(item);
      if (p<0){
        p = -new_priority;
      }
      else{
        p = new_priority;
      }
      m_q.m_bins[bin_index].update(item, p);
      m_q.m_mutexs[bin_index].unlock();
    }

    /**
     * Changes the priority to "zero"
     */
    inline void mark_visited(const T item) {
      update(item, DBL_MIN);
    }

    void remove(const T &item) {
      m_q.remove(item);
    }
  }; // end of binned scheduling queue


} // end of namespace

#include <sill/macros_undef.hpp>
#endif

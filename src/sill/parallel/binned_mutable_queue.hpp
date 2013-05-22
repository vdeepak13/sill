#ifndef BINNED_MUTABLE_QUEUE_HPP
#define BINNED_MUTABLE_QUEUE_HPP

#include <iostream>
#include <iterator>
#include <map>
#include <cassert>

#include <sill/datastructure/mutable_queue.hpp>
#include <sill/parallel/pthread_tools.hpp>

// This should come last
#include <sill/macros_def.hpp>

namespace sill {
  // Predecleration
  template<typename T>
  class binned_scheduling_queue;

  /**
   * \file binned_mutable_queue.hpp
   * 
   * \class binned_mutable_queue This is a thread safe priority queue
   * To increase performance, the queue is randomly broken up into N
   * priority queues. So to get the top element, (N-1) comparisons and
   * N mutex lock are necesary. However, promotion and updates can be
   * performed with only one lock.
   *
   * This therefore sacrifices performance of the top() and pop()
   * functions, for increased parallelism of element updates.  Do note
   * that element insertions and removals still locks the entire
   * structure.
   *
   * This design inherently requires the use of a map structure to
   * know where each element is and this map structure becomes the
   * choke point for synchronization.
   *
   * constraints: Priorities must be positive
   *
   * There are a number of functions suffixed with "unsafe"
   *   These functions are unsafe but potentially faster than their 
   *   safe versions. 
   *   The "unsafe" functions are only safe if the following conditions 
   *   are satisfied
   *     - the caller can guarantee that no insertion and deletion \n
   *       operations can happen at the same time.
   *     - the item is guaranteed to exist
   */
  template <typename T,typename Priority>
  class binned_mutable_queue {
    friend class binned_scheduling_queue<T>;
  private:
    typedef std::map<T, int> inverse_map_type;
    typedef sill::mutable_queue<T, Priority> bin_type;
    typedef std::vector<bin_type> bins_type;
   
    //! The N queues go here
    bins_type m_bins;
  
    //! synchronizes the N queues
    std::vector<spinlock> m_mutexs;
  
    //! This mutex synchronizes the inverse map and the m_count
    rwlock masterlock;

    //! Maps an element to the index of the queue it is stored in
    inverse_map_type m_inverse_map;
  
    //! Number of elements stored
    size_t m_count;
  
    int inverse_map(const T &item) const {
      masterlock.readlock();
      typename inverse_map_type::const_iterator i = m_inverse_map.find(item);
      if (i == m_inverse_map.end()) {
        masterlock.unlock();
        return -1;
      }
      int bin_index = (*i).second;
      masterlock.unlock();
      return bin_index;
    }
  
    size_t inverse_map_unsafe(const T &item){
      return m_inverse_map[item];
    }
  
  public:

    /** 
     * Constructor. Creates the binned mutable queue with bin_count
     * bins
     */
    binned_mutable_queue(const size_t bin_count) { 
      init(bin_count);
    }

    ~binned_mutable_queue() {
      clear();
    }

    /**
     * Removes the item with maximum priority from the queue, and
     * returns it as a pair with its priority.  if queue is empty, a
     * pair with negative priority will be returned
     */
    std::pair<T, Priority> pop() {
      // if queue is empty return
      bool noresult = true;
      std::pair<T, Priority> res;
      while(noresult) {
        if (empty()) {
          res.second = -1;
          return res;
        }
			
        // I must lock this early otherwise there will be a race
        // condition in between getting the top element and removing
        // it from the inverse_map
			
        int max_index = -1;
        Priority max_priority = 0;
        for(size_t i = 0; i < m_bins.size(); ++i) {
          if(m_bins[i].empty() == false && 
             m_bins[i].top().second > max_priority) {
            max_index = i; max_priority = m_bins[i].top().second;
          }
        }
        if (max_index >= 0) {
          masterlock.writelock();
          m_mutexs[max_index].lock();
          if (m_bins[max_index].empty() == false) {
            res = m_bins[max_index].pop();
            noresult = false;
            m_mutexs[max_index].unlock();
            m_inverse_map.erase(res.first);
            m_count--;
            masterlock.unlock();
          }
          else {
            m_mutexs[max_index].unlock();
            masterlock.unlock();
          }
        }
      }
      return res;
    }

  
  
    //! Returns the weight associated with a key
    Priority get(const T &item) const {
      int bin_index = inverse_map(item);
      if (bin_index<0) return -1;
      m_mutexs[bin_index].lock();
      if (!m_bins[bin_index].contains(item)) {
        m_mutexs[bin_index].unlock();
        return -1;
      }
      double res = m_bins[bin_index].get(item);
      m_mutexs[bin_index].unlock();
      return res;
    }
  
    /**  
     * Returns the weight associated with a key.  Unsafe Version. \see
     * binned_mutable_queue Note that this function is not const.
     */
    Priority get_unsafe(const T &item) {
      int bin_index = inverse_map_unsafe(item);
      if (bin_index<0) return -1;
      m_mutexs[bin_index].lock();
      if (!m_bins[bin_index].contains(item)) {
        m_mutexs[bin_index].unlock();
        return -1;
      }
      double res = m_bins[bin_index].get(item);
      m_mutexs[bin_index].unlock();
      return res;
    }
  


    /** 
     * Accesses the item with maximum priority in the queue.  If queue
     * is empty, returns an element with negative priority
     */
    const std::pair<T, Priority> top() const {
      // if queue is empty return
      bool noresult = true;
      std::pair<T, Priority> res;
      while(noresult) {
        if (empty()) {
          res.second = -1;
          return res;
        }
			
        // I must lock this early otherwise there will be a race
        // condition in between getting the top element and removing
        // it from the inverse_map
			
        int max_index = -1;
        Priority max_priority = 0;
        for(size_t i = 0; i < m_bins.size(); ++i) {
          if(m_bins[i].empty() == false && 
             m_bins[i].top().second > max_priority) {
            max_index = i; max_priority = m_bins[i].top().second;
          }
        }
        if (max_index >= 0) {
          m_mutexs[max_index].lock();
          if (m_bins[max_index].empty() == false) {
            res = m_bins[max_index].top();
            noresult = false;
            m_mutexs[max_index].unlock();
          }
          else {
            m_mutexs[max_index].unlock();
          }
        }
        if (noresult) sched_yield();
      }
      return res;
    }
  
    /** 
     * Accesses the item with maximum priority in the queue.  If queue
     * is empty, returns an element with negative priority Unsafe
     * Version. \see binned_mutable_queue
     */
    const std::pair<T, Priority>& top_unsafe() const {
      int max_index = -1;
      Priority max_priority = 0;
      while (max_index == -1) {
        for(size_t i = 0; i < m_bins.size(); ++i) {
          if(m_bins[i].empty() == false && 
            m_bins[i].top().second > max_priority) {
            max_index = i; max_priority = m_bins[i].top().second;
          }
        }
      }
      return m_bins[max_index].top();
    }

    /** 
     * Gets the maximum priority in the queue.  If queue
     * is empty, returns an element with negative priority Unsafe
     * Version. \see binned_mutable_queue
     */
    Priority top_priority_unsafe() const {
      int max_index = -1;
      Priority max_priority = 0;
      while (max_index == -1) {
        for(size_t i = 0; i < m_bins.size(); ++i) {
          if(m_bins[i].empty() == false && 
            m_bins[i].top().second > max_priority) {
            max_index = i; max_priority = m_bins[i].top().second;
          }
        }
      }
      return max_priority;
    }

  
    /** 
     * Clears the queue and recreate the binned mutable queue with
     * bin_count bins. This function is unsynchronized.
     */
    void init(const size_t bin_count) {
      clear();
      m_bins.resize(bin_count);
      m_mutexs.resize(bin_count);
      m_count = 0;
    }

    //! Returns the number of elements in the heap.
    size_t size() const { return m_count; }

    //! Returns true iff the queue is empty.
    bool empty() const { return size() == 0; }

    //! Returns true if the queue contains the given value
    bool contains(const T& item) const { 
      masterlock.readlock();
      bool retval = m_inverse_map.find(item) != m_inverse_map.end();
      masterlock.unlock();
      return retval;
    }

    /** 
     * Returns true if the queue contains the given value Unsafe
     * Version. \see binned_mutable_queue
     */
    bool contains_unsafe(const T& item) const { 
      bool retval = m_inverse_map.find(item) != m_inverse_map.end();
      return retval;
    }

    /** 
     * Enqueues a new item in the queue.  If item already exists,
     * update its priority
     */
    void push(const T &item, const Priority &priority) {
      if (contains(item)) update(item, priority);
      size_t bin_index = rand() % m_bins.size();
    
      masterlock.writelock();
      m_mutexs[bin_index].lock();
      m_bins[bin_index].push(item, priority);
      m_mutexs[bin_index].unlock();

      m_inverse_map[item] = bin_index;
      m_count++;
      masterlock.unlock();
    }


    /** 
     * Clears all the values (equivalent to stl clear) This function
     * is unsynchronized.
     */
    void clear() {
      for (size_t i = 0; i < m_bins.size(); ++i) m_bins[i].clear();
      m_inverse_map.clear();
      m_count = 0;
    }

    //! Remove an item from the queue
    void remove(const T &item) {
      // Verify that the item is currently in the queue
      int bin_index = inverse_map(item);
      if (bin_index < 0) return;
      // I must lock this early otherwise there is
      // a race between removing the item from the queue
      // and updating the inverse map
      masterlock.writelock();    
      m_mutexs[bin_index].lock();
      if (m_bins[bin_index].contains(item)) m_bins[bin_index].remove(item);
      m_mutexs[bin_index].unlock();
    
      m_inverse_map.erase(item);
      m_count--;
      masterlock.unlock();
    }
  
    //! returns the priority of the top element
    inline double top_priority() {
      return top().second;
    }

    //! returns the priority of element id
    inline Priority operator[](const T &id) {
      Priority p = get(id);
      return p;
    }
  
    /** 
     * Increases the priority of 'item' to 'new_priority' Priority is
     * updated only if the old priority is lower than the new priority
     */
    inline void promote(const T &item, const Priority &new_priority) {  
      int bin_index = inverse_map(item);
      if (bin_index<0) return;
      m_mutexs[bin_index].lock();
      if (m_bins[bin_index].contains(item)) {
        Priority p = m_bins[bin_index].get(item);
        p = std::max(p,new_priority);
        m_bins[bin_index].update(item, p);
      }
      m_mutexs[bin_index].unlock();
    }

    //! Changes the priority of 'item' to 'new_priority'
    inline void update(const T &item, const Priority &new_priority) {
      int bin_index = inverse_map(item);
      if (bin_index<0) return;
      m_mutexs[bin_index].lock();
      if (m_bins[bin_index].contains(item)) {
        m_bins[bin_index].update(item, new_priority);
      }
      m_mutexs[bin_index].unlock();
    }
  
    /** 
     * Increases the priority of 'item' to 'new_priority' Priority is
     * updated only if the old priority is lower than the new priority
     * Unsafe Version. \see binned_mutable_queue
     */
    inline void promote_unsafe(const T &item, const Priority &new_priority) {  
      int bin_index = inverse_map_unsafe(item);
      if (bin_index<0) return;
      m_mutexs[bin_index].lock();
      Priority p = m_bins[bin_index].get(item);
      p = std::max(p,new_priority);
      m_bins[bin_index].update(item, p);
      m_mutexs[bin_index].unlock();
    }
  
    /** 
     * Changes the priority of 'item' to 'new_priority' Unsafe
     * Version. \see binned_mutable_queue
     */
    inline void update_unsafe(const T &item, const Priority &new_priority) {
      int bin_index = inverse_map_unsafe(item);
      if (bin_index<0) return;
      m_mutexs[bin_index].lock();
      m_bins[bin_index].update(item, new_priority);
      m_mutexs[bin_index].unlock();
    }
  }; // end of binned mutable queue
 
} // end of namespace

#include <sill/macros_undef.hpp>
#endif

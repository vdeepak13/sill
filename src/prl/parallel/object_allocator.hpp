#ifndef OBJECT_ALLOCATOR_HPP
#define OBJECT_ALLOCATOR_HPP
#include <pthread.h>
#include <prl/parallel/pthread_tools.hpp>

#include <prl/macros_def.hpp>
namespace prl {
/**
  * \class object_allocator 
  * 
  * Implements a small pool of objects which users can checkin and
  * checkout.  Checkedin objects are then reused.  This is useful for
  * situations where you need an object for temporary storage, but do
  * not desire to pay the penalty for a malloc.
  *
  * This pool is synchronized and is thread-safe
  */
template<typename T> 
class object_allocator{
private:
  spinlock listlock;
  std::set<T*> freelist;
  std::set<T*> usedlist;
public:
  object_allocator() {};  
  
  ~object_allocator() {
    // clear the free list
    for(typename std::set<T*>::iterator t = freelist.begin();
        t != freelist.end();
        ++t) {
      delete *t;
    }
  }
  
  /**
    * checks out an object from the pool. An object will not be
    * simultaneously given to more than one user.  If the free list
    * is empty, the default constructor for the object is used to
    * create new objects.
    */
  T* checkout() {
    T* ret;
    listlock.lock();
    if (freelist.size() > 0) {
      ret = *(freelist.begin());
      freelist.erase(ret);
    }
    else {
      ret = new T;
    }
    usedlist.insert(ret);
    listlock.unlock();
    return ret;
  }

  /**
    * Returns an object from the pool. This object can then by given
    * to other users 
    *
    * BUGBUG: This is a linear time operations.  I don't think we
    * want that
    */
  void checkin(const T* f) {
    listlock.lock();
    // Linear time operations
    usedlist.erase(f);
    freelist.insert(f);
    listlock.unlock();
  }
}; // End of object_allocator







/**
  Alternate version of the object_allocator
  which works by allocating an object per thread. (using the thread's id
  to differentiate) instead of sharing among a common pool.
  
  The synchronization method used here is not 100% safe as it makes some 
  assumptions about how the STL map works. Portability is not guaranteed.
*/
template<typename T> 
class object_allocator_mapbased{
private:
  std::map<pthread_t, T*> objectpool;
  mutex m; 
public:
  object_allocator_mapbased() {}
  ~object_allocator_mapbased() {
    // clear the free list
    for(typename std::map<pthread_t, T*>::iterator t = objectpool.begin();
        t != objectpool.end();
        ++t) {
      delete t->second;
    }
  }
  T* checkout() {
    pthread_t id = pthread_self();
    // try an unsafe find
    typename std::map<pthread_t, T*>::iterator i = objectpool.find(id);
    if (i == objectpool.end()) {
      // unsafe find failed. Lock and try a safe find
      m.lock();
      i = objectpool.find(id);
      // still not found. therefore it doesn't exist. insert it
      if (i == objectpool.end()) {
        T* ret = new T;
        objectpool[id] = ret;
        m.unlock();
        return ret;
      }
      else {
        T* ret = i->second;
        m.unlock();
        return ret;
      }
      
      
    }
    else {
      return i->second;
    }
  }
  // no code required for checkin, since there is no common pool
  void checkin(const T* f) {}
}; // End of object_allocator








/**
  Yet another version of the object_allocator
  which works by allocating an object per thread and storing the pointer
  in the TLS (Thread Local Storage)
  
  As opposed to object_allocator_mapbased, this is 100% threadsafe
  but requires support from pthreads
*/
template<typename T> 
class object_allocator_tls{
private:
  pthread_key_t tlskey;
  
  static void tls_destructor(void* v){ 
    if (v != NULL) {
      delete (T*)v;
    }
  }
  
public:
  object_allocator_tls() {
    assert(pthread_key_create(&tlskey, object_allocator_tls::tls_destructor)==0);
  }
  
  // no destructor necessary, pthread will call the destructor function
  // automatically
  ~object_allocator_tls() {}
  
  T* checkout() {
    T* ret = (T*)(pthread_getspecific(tlskey));
    // we are guaranteed that if we don't set a value to the TLS, 
    // it will be NULL. Therefore if it is NULL, we haven't created the 
    // object yet
    if (ret == NULL) {
      ret = new T;
      pthread_setspecific(tlskey, ret);
    }
    return ret;
  }
  // no code required for checkin, since there is no common pool
  void checkin(const T* f) {}
}; // End of object_allocator



/**
  Yet another version of the object_allocator
  which works by allocating any number of objects per thread and storing the pointer
  in the TLS (Thread Local Storage)
  
  As opposed to object_allocator_mapbased, this is 100% threadsafe
  but requires support from pthreads
*/
template<typename T> 
class multi_object_allocator_tls{
private:
  pthread_key_t tlskey;
  
  static void tls_destructor(void* v){
    pool* p = (pool*)v;
    if (p == NULL) return;
    foreach(T* t, p->free) {
      delete t;
    }
    delete p;
  }
public:

  struct pool{
    std::list<T*> free;
    std::set<T*> inuse;
  };

  multi_object_allocator_tls() {
    assert(pthread_key_create(&tlskey, multi_object_allocator_tls::tls_destructor)==0);
  }
  
  // no destructor necessary, pthread will call the destructor function
  // automatically
  ~multi_object_allocator_tls() {}
  
  T* checkout() {
    pool* p = (pool*)(pthread_getspecific(tlskey));
    // we are guaranteed that if we don't set a value to the TLS, 
    // it will be NULL. Therefore if it is NULL, we haven't created the 
    // object yet
    if (p == NULL) {
      p = new pool;
      pthread_setspecific(tlskey, p);
    }
    
    if (p->free.size() >0) {
      T* t = p->free.front();
      p->free.pop_front();
      p->inuse.insert(t);
      return t;
    }
    else {
      T* t = new T;
      p->inuse.insert(t);
      return t;
    }
  }
  // no code required for checkin, since there is no common pool
  void checkin(const T* f) {
    pool* p = (pool*)(pthread_getspecific(tlskey));
    p->inuse.erase(const_cast<T*>(f));
    p->free.push_front(const_cast<T*>(f));
    
  }
}; // End of object_allocator
};
#include <prl/macros_undef.hpp>
#endif

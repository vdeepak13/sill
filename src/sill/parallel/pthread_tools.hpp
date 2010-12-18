#ifndef PTHREAD_TOOLS_HPP
#define PTHREAD_TOOLS_HPP

#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <sched.h>
#include <list>
#include <iostream>
#include <sys/time.h>
#include <cassert>

/**
 * \file pthread_tools.hpp A collection of utilities for threading
 */
namespace sill {
  /**
   * \class runnable Base class for defining a threaded function call.
   * A pointer to an instance of this class is passed to
   * thread_group. When the thread starts the run() function will be
   * called.
   */
  class runnable {
  public:
    //! The function that is executed when the thread starts 
    virtual void run() = 0;
    virtual ~runnable() {};
  };


  /**
   * \class thread is a basic thread which is runnable but also has
   * the ability to be started atomically by invoking start. To use
   * this class simply extend it, implement the runnable method and 
   * and invoke the start method.
   */
  class thread : public runnable {
  private:
    //! Little helper function used to launch runnable objects
    static void* invoke(void *r) { 
      static_cast<runnable*>(r)->run();
      pthread_exit(NULL);
    }

    //! The size of the internal stack for this thread
    size_t m_stack_size;

    //! The internal pthread object
    pthread_t m_p_thread;

    /**
     * The object this thread runs.  This object must stay in scope
     * for the duration of this thread 
     */
    runnable* m_runnable;
    
    bool m_active;

  public:
    
    /**
     * Creates a thread that either runs the passed in runnable object
     * or otherwise invokes itself.
     */
    thread(runnable* obj = NULL) : m_runnable(obj), m_active(false) { 
      // Calculate the stack size in in bytes;
      const int BYTES_PER_MB = 1048576; 
      const int DEFAULT_SIZE_IN_MB = 8;
      m_stack_size = DEFAULT_SIZE_IN_MB * BYTES_PER_MB;
    }

    /**
     * execute this function to spawn a new thread running the run
     * routine provided by runnable:
     */
    void start(int priority = 0) {
      // fill in the thread attributes
      pthread_attr_t attr;
      pthread_attr_init(&attr);
      pthread_attr_setstacksize(&attr, m_stack_size);
      pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);  

      sched_param param;
      if (priority != 0) {
        pthread_attr_getschedparam(&attr, &param);
        param.sched_priority = priority;
        pthread_attr_setschedparam(&attr, &param);
      }
      // If no runnable object was passed in then this thread will try
      // and run itself.
      if(m_runnable == NULL) m_runnable = this;

    
      // Launch the thread.  Effectively this creates a new thread
      // which calls the invoke function passing in a pointer to a
      // runnable object
      int rc = pthread_create(&m_p_thread, 
                              &attr, 
                              invoke,  
                              static_cast<void*>(m_runnable) );
      m_active = true;
      if(rc) {
        std::cout << "Major error in thread_group.launch" << std::endl;
        exit(EXIT_FAILURE);
      }
      // destroy the attribute object
      pthread_attr_destroy(&attr);
    }

    /**
     * Same as start() except that you can specify a CPU on which to
     * run the thread.  This only currently supported in Linux and if
     * invoked on a non Linux based system this will be equivalent to
     * start()
     */
    void start(size_t cpu_id){
      // if this is not a linux based system simply invoke start and return;
#ifndef __linux__
      start();
      return;
#else
      // At this point we can assert that this is a linux system
      if (cpu_id >= cpu_count() && cpu_count() > 0) {
        // definitely invalid cpu_id
        std::cout << "Invalid cpu id passed on thread_ground.launch()" 
                  << std::endl;
        std::cout << "CPU " << cpu_id << "requested, but only " 
                  << cpu_count() << " CPUs available" << std::endl;
        exit(EXIT_FAILURE);
      }
      
      // fill in the thread attributes
      pthread_attr_t attr;
      pthread_attr_init(&attr);
      pthread_attr_setstacksize(&attr, m_stack_size);
      pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);  
      
      // Set Processor Affinity masks (linux only)
      cpu_set_t cpu_set;    
      CPU_ZERO(&cpu_set);
      CPU_SET(cpu_id % CPU_SETSIZE, &cpu_set);
      pthread_attr_setaffinity_np(&attr, sizeof(cpu_set), &cpu_set);

      // If no runnable object was passed in then this thread will try
      // and run itself.
      if(m_runnable == NULL) m_runnable = this;
      
      // Launch the thread
      int rc = pthread_create(&m_p_thread, 
                              &attr, 
                              invoke,
                              static_cast<void*>(m_runnable) );
      m_active = true;
      if(rc) {
        std::cout << "Major error in thread_group.launch" << std::endl;
        std::cout << "pthread_create() returned error " << rc << std::endl;
        exit(EXIT_FAILURE);
      }
      // destroy the attribute object
      pthread_attr_destroy(&attr);
#endif
    } // end of start(size_t cpu_id)


    /**
     * Join the calling thread with this thread.
     */
    void join() {
      if(this == NULL) {
        std::cout << "Failure on join()" << std::endl;
        exit(EXIT_FAILURE);
      }
      join(*this);
    }

    bool active() const { return m_active; }

    /**
     * This static method joins the invoking thread with the other
     * thread object.  This thread will not return from the join
     * routine until the other thread complets it run.
     */
    static void join(thread& other) {

      void *status;
      // joint the first element
      int rc = 0;
      if(other.active())
        rc = pthread_join( other.m_p_thread, &status);
      if(rc) {
        std::cout << "Major error in join" << std::endl;
        std::cout << "pthread_join() returned error " << rc << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    
    /**
     * Return the number processing units (individual cores) on this
     * system
     */
    static size_t cpu_count() {
#ifdef __linux__
      return sysconf(_SC_NPROCESSORS_CONF);
#else
      return 0;
#endif
    }


    /**
     * Default run routine is abstract and should be implemented by a
     * child class.
     */
    virtual void run() {
      std::cout << "Function run() is not implemented." << std::endl;
      exit(EXIT_FAILURE);
    }

    
    virtual ~thread() {   }
  }; // End of class thread



  /**
   * \class thread_group Manages a collection of threads
   */
  class thread_group {
    std::list< thread > m_threads;
  public:
    /** 
     * Initializes a thread group. 
     */
    thread_group() { }

    /** 
     * Launch a single thread which calls r->run() No CPU affinity is
     * set so which core it runs on is up to the OS Scheduler
     */
    void launch(runnable* r) {
      if(r == NULL) {
        std::cout << "Launching a NULL pointer." << std::endl;
        exit(EXIT_FAILURE);
      }
      // Create a thread object
      thread local_thread(r);
      local_thread.start();
      // keep a local copy of the thread
      m_threads.push_back(local_thread);
    } // end of launch

    /**
     * Launch a single thread which calls r->run() Also sets CPU
     *  Affinity
     */
    void launch(runnable* r, size_t cpu_id) {
      if(r == NULL) {
        std::cout << "Launching a NULL pointer." << std::endl;
        exit(EXIT_FAILURE);
      }
      // Create a thread object
      thread local_thread(r);
      local_thread.start(cpu_id);
      // keep a local copy of the thread
      m_threads.push_back(local_thread);
    } // end of launch

    //! Waits for all threads to complete execution
    void join() {
      while(!m_threads.empty()) {
        m_threads.front().join(); // Join the first thread
        m_threads.pop_front(); // remove the first element
      }
    }

    //! Destructor. Waits for all threads to complete execution
    ~thread_group(){
      join();
    }

  }; // End of thread group


  /**
   * \class mutex 
   * 
   * Wrapper around pthread's mutex On single core systems mutex
   * should be used.  On multicore systems, spinlock should be used.
   */
  class mutex {
  private:
    // mutable not actually needed
    mutable pthread_mutex_t m_mut;
  public:
    mutex() { pthread_mutex_init(&m_mut, NULL); }
    inline void lock() const { 
      pthread_mutex_lock( &m_mut  ); 
    }
    inline void unlock() const {
      pthread_mutex_unlock( &m_mut ); 
    }
    inline bool try_lock() const {
      return pthread_mutex_trylock( &m_mut ) == 0;
    }
    ~mutex(){ pthread_mutex_destroy( &m_mut ); }
    friend class conditional;
  }; // End of Mutex

#if _POSIX_SPIN_LOCKS >= 0
  // We should change this to use a test for posix_spin_locks eventually
  
  // #ifdef __linux__
  /**
   * \class spinlock
   * 
   * Wrapper around pthread's spinlock On single core systems mutex
   * should be used.  On multicore systems, spinlock should be used.
   * If pthread_spinlock is not available, the spinlock will be
   * typedefed to a mutex
   */
  class spinlock {
  private:
    // mutable not actually needed
    mutable pthread_spinlock_t m_spin;
  public:
    spinlock () { pthread_spin_init(&m_spin, PTHREAD_PROCESS_PRIVATE); }
  
    inline void lock() const { 
      pthread_spin_lock( &m_spin  ); 
    }
    inline void unlock() const {
      pthread_spin_unlock( &m_spin ); 
    }
    inline bool try_lock() const {
      return pthread_spin_trylock( &m_spin ) == 0;
    }
    ~spinlock(){ pthread_spin_destroy( &m_spin ); }
    friend class conditional;
  }; // End of spinlock
#define SPINLOCK_SUPPORTED 1
#else
  //! if spinlock not supported, it is typedef it to a mutex.
  typedef mutex spinlock;
#define SPINLOCK_SUPPORTED 0
#endif


  /**
   * \class conditional
   * Wrapper around pthread's condition variable
   */
  class conditional {
  private:
    mutable pthread_cond_t  m_cond;
  public:
    conditional() { pthread_cond_init(&m_cond, NULL); }
    inline void wait(const mutex& mut) const {
      pthread_cond_wait(&m_cond, &mut.m_mut);
    }
    inline int timedwait(const mutex& mut, int sec) const {
      struct timespec timeout;
      struct timeval tv;
      struct timezone tz;
      gettimeofday(&tv, &tz);
      timeout.tv_nsec = 0;
      timeout.tv_sec = tv.tv_sec + sec;
      return pthread_cond_timedwait(&m_cond, &mut.m_mut, &timeout);
    }
    inline void signal() const { pthread_cond_signal(&m_cond);  }
    inline void broadcast() const { pthread_cond_broadcast(&m_cond); }
    ~conditional() { pthread_cond_destroy(&m_cond); }
  }; // End conditional

  /**
   * \class semaphore
   * Wrapper around pthread's semaphore
   */
  class semaphore {
  private:
    mutable sem_t  m_sem;
  public:
    semaphore() { sem_init(&m_sem, 0,0); }
    inline void post() const { sem_post(&m_sem);  }
    inline void wait() const { sem_wait(&m_sem); }
    ~semaphore() { sem_destroy(&m_sem); }
  }; // End semaphore


  /**
   * \class rwlock
   * Wrapper around pthread's rwlock
   */
  class rwlock {
  private:
    mutable pthread_rwlock_t m_rwlock;
  public:
    rwlock() { pthread_rwlock_init(&m_rwlock, NULL); }
    ~rwlock() { pthread_rwlock_destroy(&m_rwlock); }
    inline void readlock() const {  pthread_rwlock_rdlock(&m_rwlock);  }
    inline void writelock() const {  pthread_rwlock_wrlock(&m_rwlock);  }
    inline void unlock() const { pthread_rwlock_unlock(&m_rwlock); }
  }; // End rwlock


#ifdef __linux__
  /**
   * \class barrier
   * Wrapper around pthread's barrier
   */
  class barrier {
  private:
    mutable pthread_barrier_t m_barrier;
  public:
    barrier(size_t numthreads) { pthread_barrier_init(&m_barrier, NULL, numthreads); }
    ~barrier() { pthread_barrier_destroy(&m_barrier); }
    inline void wait() const { pthread_barrier_wait(&m_barrier); }  
  };

#else
  /**
   * \class barrier
   * Wrapper around pthread's barrier
   */
  class barrier {
  private:
    // mutable pthread_barrier_t m_barrier;
  public:
    barrier(size_t numthreads) {
      std::cout << "p_thread_barrier not supported on this platform" << std::endl;
      assert(false);
      // pthread_barrier_init(&m_barrier, NULL, numthreads); 
    }
    ~barrier() { 
      std::cout << "p_thread_barrier not supported on this platform" << std::endl;
      assert(false);  
      // pthread_barrier_destroy(&m_barrier); 
    }
    inline void wait() const { 
      std::cout << "p_thread_barrier not supported on this platform" << std::endl;
      assert(false);  
      // pthread_barrier_wait(&m_barrier); 
    }  
  };
#endif




}; // End Namespace

#endif

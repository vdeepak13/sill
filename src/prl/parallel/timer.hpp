#ifndef TIMER_HPP
#define TIMER_HPP

#include <sys/time.h>
#include <stdio.h>

#include <iostream>

namespace prl {
  /**
   *   \class timer A simple class that can be used for
   *   benchmarking/timing up to microsecond resolution.
   */
  class timer {
  private:
    timeval start_time_;   
  public:
    timer() { }
  
    //! Starts the timer. 
    void start() { gettimeofday(&start_time_, NULL); }
  
    /** 
     * Returns the number of seconds since start() was called Behavior
     *  is undefined if start() was not called before calling
     *  current_time()
     */
    double current_time() const {
      timeval current_time;
      gettimeofday(&current_time, NULL);
      double answer = 
        (current_time.tv_sec + current_time.tv_usec/1.0E6) -
        (start_time_.tv_sec + start_time_.tv_usec/1.0E6);
      return answer;
    }
  }; // end of Timer

  /** 
   * Convenience function. Allows you to call "cout << ti" where ti is
   * a timer object and it will print the number of seconds elapsed
   * since ti.start() was called.
   */
  template<typename Char, typename Traits>
  std::basic_ostream<Char, Traits>&
  operator<<(std::basic_ostream<Char, Traits>& out, const prl::timer& t) {
    out << t.current_time();
    return out;
  }

} // end of prl namespace

#endif

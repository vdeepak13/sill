/*
 * belief_convergence_measures.hpp
 *
 *  Created on: Mar 11, 2009
 *      Author: antonc
 */

#ifndef BP_CONVERGENCE_MEASURES_HPP
#define BP_CONVERGENCE_MEASURES_HPP

#include <boost/timer.hpp>

namespace sill {

  class runnable_double_arg{
  public:
    virtual ~runnable_double_arg() { }
    virtual void run(double d)=0;
  };

  //! interface of classes that determine convergence
  class residual_splash_convergence_measure{
  private:
    //! residual convergence bound
    const double residual_bound_;

    //! determines if start() has been called, need to enforce it for timers
    bool started_;
  public:

    residual_splash_convergence_measure(double residual_bound) :
      residual_bound_(residual_bound), started_(false){
      assert(residual_bound_ > 0);
    }

    //! returns true iff the algorithm converged after this update
    virtual bool is_converged( double max_residual,
                                  size_t last_iteration_updates){
      assert(started_);

      if(max_residual < residual_bound_)
        return true;
      else return false;
    }

    double residual_bound(){ return residual_bound_; };

    //! use this to initialize timers in the measures that use them
    virtual void start(){
      assert(!started_);
      started_ = true;
    };

    virtual ~residual_splash_convergence_measure(){};
  };


  class updates_count_convergence_measure :
    public residual_splash_convergence_measure {

  private:
    //! max updates allowed. negative means unlimited
    int max_updates_, total_updates_;

  public:
    updates_count_convergence_measure(double residual_bound, int max_updates) :
      residual_splash_convergence_measure(residual_bound),
      max_updates_(max_updates), total_updates_(0){};

    virtual bool is_converged( double max_residual,
                                  size_t last_iteration_updates){

      if(residual_splash_convergence_measure::is_converged(
            max_residual, last_iteration_updates))
        return true;

      total_updates_ += last_iteration_updates;
      if( max_updates_ > 0 && total_updates_ > max_updates_)
        return true;
      else return false;
    }
  };

  class cpu_time_convergence_measure :
    public residual_splash_convergence_measure {

  private:
    //! set of time bounds
    std::vector<double> time_bounds_;

    //! current time bound index
    size_t bound_index_;

    //! cpu timer
    boost::timer* timer_;

    //! The object with a function to run on crossing the time bounds
    runnable_double_arg* time_bounds_action_;

  public:

    cpu_time_convergence_measure( double residual_bound,
                                  double time_bound,
                                  runnable_double_arg* time_bounds_action = NULL) :
        residual_splash_convergence_measure(residual_bound),
        bound_index_(0),
        timer_(NULL),
        time_bounds_action_(time_bounds_action){

      assert(time_bound > 0);
      time_bounds_.push_back(time_bound);
    }

    cpu_time_convergence_measure( double residual_bound,
                                  const std::vector<double>& time_bounds,
                                  runnable_double_arg* time_bounds_action = NULL) :
        residual_splash_convergence_measure(residual_bound),
        time_bounds_(time_bounds),
        bound_index_(0),
        timer_(NULL),
        time_bounds_action_(time_bounds_action){

      assert(time_bounds_.size() > 0);
      std::sort(time_bounds_.begin(), time_bounds_.end());
      assert(time_bounds_[0] > 0);
     }

    virtual void start(){
      residual_splash_convergence_measure::start();
      //restart the timer
      timer_ = new boost::timer();
    }

    virtual bool is_converged( double max_residual,
                                  size_t last_iteration_updates){
      if(residual_splash_convergence_measure::is_converged(
            max_residual, last_iteration_updates)){
        do
          time_bounds_action_->run(time_bounds_[bound_index_]);
        while(++bound_index_ < time_bounds_.size());

        return true;
      }

      if( timer_->elapsed() > time_bounds_[bound_index_]){
        if(time_bounds_action_ != NULL)
          time_bounds_action_->run(time_bounds_[bound_index_]);

        if(++bound_index_ == time_bounds_.size())
          return true;
        else
          return false;
      }

      return false;
    }

    virtual ~cpu_time_convergence_measure(){
      if(timer_ != NULL)
        delete timer_;
    }
  };
}


#endif /* BELIEF_CONVERGENCE_MEASURES_HPP_ */

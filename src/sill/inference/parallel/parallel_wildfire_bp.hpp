#ifndef PARALLEL_WILDFIRE_BP_HPP
#define PARALLEL_WILDFIRE_BP_HPP


// STL includes
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <limits>

// PRL Includes
#include <sill/inference/interfaces.hpp>
#include <sill/model/factor_graph_model.hpp>
#include <sill/factor/util/norms.hpp>
#include <sill/math/gdl_enum.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/factor/log_table_factor.hpp>
#include <sill/datastructure/mutable_queue.hpp>
#include <sill/model/factor_graph_partitioning.hpp>
#include <sill/datastructure/circular_queue.hpp>
// Pthread tools
#include <sill/parallel/pthread_tools.hpp>


// This include should always be last
#include <sill/macros_def.hpp>

namespace sill {

  template<typename F>
  class parallel_wildfire_bp:
          public factor_graph_inference<factor_graph_model<F> > {
    ///////////////////////////////////////////////////////////////////////
    // typedefs
    public:
      typedef factor_graph_model<F>     factor_graph_type;
      typedef typename factor_graph_type::variable_type    variable_type;
      typedef typename factor_graph_type::vertex_type      vertex_type;

      ///////////////////////////////////////////////////////////////////////
      // Data members
    private:
      //! pointer to the factor graph
      factor_graph_type* factor_graph_;

      //! A priority queue of vertices for each processor
      // mutable_queue<size_t, double> schedule_;
      std::queue<size_t> schedule_;
      std::vector<double> residuals_;
      //! The lock for each queue
      spinlock schedulelock_;

      //! All threads will terminate when this is set to true
      bool finished_;
      
      //! A list of all messages. To read/write a message from src->dest
      //! you should lock the belief at the destination vertex.
      std::vector<std::map<size_t, F> > messages_;

      //! A list of all vertex beliefs. You shoud lock the belief before
      //! reading or writing
      std::vector<F> beliefs_;
      std::vector<spinlock> beliefslock_;

      //! Number of processors
      size_t numprocs_;
      
      //! convergence bound
      double bound_;

      //! level of damping 1.0 is fully damping and 0.0 is no damping
      double damping_;

      //! the commutative semiring for updates (typically sum_product)
      commutative_semiring csr_;

      bool wakeupbroadcasted_;
      double timeout_;
      
      size_t numupdates_;
      size_t numops_;
      size_t numedgeupdates_;
      double runtime_;

    public:
      /**
        Blf Splash processing thread
      */
      class wildfire_thread : public thread {
        public:
          /// a reference to the parent class
          parallel_wildfire_bp &owner_;

          /// the index of the scheduling queue I am using
          size_t scheduleid_;
          
          size_t numupdates_;
          size_t numops_;
          size_t numedgeupdates_;
          
          std::queue<size_t> *queue_;
          
          wildfire_thread(parallel_wildfire_bp &owner, size_t scheduleid,
                            std::queue<size_t> *cq)
                                    :owner_(owner),scheduleid_(scheduleid),
                                     queue_(cq) {}

          ~wildfire_thread() {}
          
          void run() {
            numupdates_ = 0;
            numops_ = 0;
            numedgeupdates_ = 0;
          
            // start a timer to figure when to print
            timer ti;
            ti.start();
            double lasttime = ti.current_time();
            
            std::cout << "Starting Run in worker" << std::endl;
            while(owner_.converged(scheduleid_) == false) {
              owner_.schedulelock_.lock();
              size_t vid = queue_->front();
              queue_->pop();
              double lastresidual = owner_.residuals_[vid];
              owner_.residuals_[vid] = 0;
              owner_.schedulelock_.unlock();
              vertex_type v = owner_.factor_graph_->id2vertex(vid);
              owner_.send_messages(v);
              ++numupdates_;
              numedgeupdates_ += owner_.factor_graph_->num_neighbors(vid);
              numops_ += owner_.factor_graph_->work_per_update(v);
              if(numupdates_ % 50000 == 0) {
                if (owner_.timeout_ > 0 && ti.current_time() >= owner_.timeout_) break;
                if (scheduleid_ == 0) {
                  std::cout << scheduleid_ << ": " << lastresidual<<": "<<numupdates_ << std::endl;
                  lasttime = ti.current_time();
                }
              }
            }
          }
      }; // end wildfire_thread

      /**
      * Create an engine (without allocating messages) with the factor
      * graph, splash size, convergence bound, and damping.
      */
      parallel_wildfire_bp():
          csr_(sum_product) {
        factor_graph_ = NULL;
        bound_ = 0.001;
        damping_ = 0.0;
      } // end parallel_msplash_bp

      void clear() {
        factor_graph_ = NULL;
        bound_ = 0.001;
        damping_ = 0.0;
        schedule_ = std::queue<size_t>();
        residuals_.clear();
        messages_.clear();
        beliefs_.clear();
        beliefslock_.clear();
      }


      void initialize(factor_graph_type* factor_graph,
                      size_t numprocs,
                      double bound,
                      double damping,
                      double timeout = 0.0) {
        timeout_ = timeout;
        numprocs_ = numprocs;
        factor_graph_ = factor_graph;
        bound_ = bound;
        damping_ = damping;
        // Clear the messages
        messages_.clear();
        messages_.resize(factor_graph_->num_vertices());
        // Allocate all messages
        beliefs_.resize(factor_graph_->num_vertices());
        beliefslock_.resize(factor_graph_->num_vertices());
        foreach(const vertex_type& u, factor_graph_->vertices()) {
          foreach(const vertex_type& v, factor_graph_->neighbors(u)) {
            F& msg = message(u.id(),v.id());
            finite_domain domain = make_domain(u.is_variable() ?
                                              &(u.variable()) :
                                              &(v.variable()));
            msg = F(domain, 1.0).normalize();
          }
          // Initialize the belief
          F& blf = beliefs_[u.id()];
          if(u.is_factor()) {
            blf = u.factor();
          } else {
            blf = F(make_domain(&(u.variable())),1.0).normalize();
          }
        }
        // Initialize the schedule
        schedule_ = std::queue<size_t>();
        finished_ = false;
        residuals_.resize(factor_graph_->num_vertices());
        size_t idx = 0;
        // double initial_residual = bound_ + (bound_*bound_);
        double initial_residual = std::numeric_limits<double>::max();
        foreach(const vertex_type& u, factor_graph_->vertices()) {
          residuals_[u.id()] = initial_residual;
          idx++;
        }

        wakeupbroadcasted_ = false;
        factor_graph_->build_work_per_update_cache();
      } // end of initialize



      // the splash thread will call this function
      // the schedule[scheduleid_] MUST BE LOCKED prior to entering this
      // function. When this function returns, it will still be locked.
      bool converged(size_t scheduleid) {
        if (finished_) return true;
        schedulelock_.lock();
        finished_ = schedule_.size() == 0; 
        schedulelock_.unlock();
        return finished_;
      }


      double loop_to_convergence() {
        finished_ = false;
        std::cout << "Started..." << std::endl;
        std::vector<wildfire_thread*> threads;
        threads.resize(numprocs_);

        for (size_t i = 0;i < factor_graph_->num_vertices(); ++i) {
          schedule_.push(i);
        }

        timer ti;
        ti.start();

        numupdates_ = 0;
        numops_ = 0;
        numedgeupdates_ = 0;
        
        for (size_t i = 0;i < numprocs_; ++i) {
          threads[i] = new wildfire_thread(*this,i,&schedule_);
          threads[i]->start();
        }

        for (size_t i = 0;i < numprocs_; ++i) {
          threads[i]->join();
          numupdates_ += threads[i]->numupdates_;
          numops_ += threads[i]->numops_;
          numedgeupdates_ += threads[i]->numedgeupdates_;
          delete threads[i];
        }
        runtime_ = ti.current_time();
        return runtime_;
      } // end of splash_to_convergence



      // Update the message changing the destination belief and priority
      // and returning the belief residual.
      inline double update_message(const vertex_type& source,
                                  const vertex_type& target,
                                  const F& new_msg,
                                   const F& damped_msg) {

        size_t targetid = factor_graph_->vertex2id(target);
        //---------------------------------------------
        // Lock the target belief and update the message
        //---------------------------------------------
        beliefslock_[targetid].lock();
        F& targetbelief = beliefs_[target.id()];
        // lock the message before the belief
        F& original_msg = message(source.id(),target.id());

        // Make a backup of the old belief
        F prevblf = targetbelief;

        // Update the new belief by dividing out the old message and
        // multiplying in the new message.
        targetbelief.combine_in(original_msg, divides_op);
        targetbelief.combine_in(new_msg, csr_.dot_op);

        // Ensure that the belief remains normalized
        targetbelief.normalize();
        // Compute the new residual
        double belief_residual = norm_1(prevblf, targetbelief);

        targetbelief.combine_in(new_msg, divides_op);
        targetbelief.combine_in(damped_msg, csr_.dot_op);
        targetbelief.normalize();
       // Update the message
        original_msg = damped_msg;

        beliefslock_[targetid].unlock();


        // update the priority
        schedulelock_.lock();

        double new_priority = residuals_[target.id()] + belief_residual;
        // For the residual to be finite
        if(!std::isfinite(new_priority)) {
          new_priority = std::numeric_limits<double>::max();
        }
        if (residuals_[target.id()] <= bound_ && new_priority > bound_) {
          schedule_.push(target.id());
        }
        residuals_[target.id()] = new_priority;

        schedulelock_.unlock();

        return new_priority;
      } // end of update_message




      inline void send_messages(const vertex_type& source) {
        // For each of the neighboring vertices (vertex_target) of
        // vertex compute the message from vertex to vertex_target
        foreach(const vertex_type &dest,
                factor_graph_->neighbors(source)) {
          send_message(source, dest);
        }
      } // end of update messages


      // Recompute the message from the source to target updating the
      // target belief and returning the change in target belief value
      inline double send_message(const vertex_type& source,
                                const vertex_type& target) {
        size_t sourceid = factor_graph_->vertex2id(source);
        size_t targetid = factor_graph_->vertex2id(target);

        // lock the belief and the incoming message to compute the cavity
        beliefslock_[sourceid].lock();
        // here we can assume that the current belief is correct
        F& blf = beliefs_[sourceid];
        // Construct the cavity
        F cavity = combine(blf, message(target.id(),source.id()), divides_op);
        // lock the message before the belief
        beliefslock_[sourceid].unlock();

        cavity.normalize();

        // Marginalize out any other variables
        finite_domain domain = make_domain(source.is_variable()?
                                        &(source.variable()) :
                                        &(target.variable()));

        F new_msg = cavity.collapse(csr_.cross_op, domain);

        // Normalize the message
        new_msg.normalize();

        // Lock the target to compute the weighted update
        beliefslock_[targetid].lock();
        F damped_msg;
        // Damp messages form factors to variables
        if(target.is_variable()) {
          damped_msg = weighted_update(new_msg,
                                    message(source.id(), target.id()),
                                    damping_);
        }
        else {
          damped_msg = new_msg;
        }
        beliefslock_[targetid].unlock();
        new_msg.normalize();

        // update the message also updating the schedule
        double belief_residual =  update_message(source, target, new_msg, damped_msg);


        return belief_residual;
      } // end of send_message







      //! get the message from source to target
      inline const F& message(const size_t source,
                              const size_t target) const {
        return safe_get(messages_[source], target);
      } // end of message

      //! get the message from source to target
      inline F& message(const size_t source,
                        const size_t target)  {
        return messages_[source][target];
      } // end of message


      /**
      * Compute the belief for a vertex
      */
      const F& belief(variable_type* variable) const{
        return beliefs_[factor_graph_->to_vertex(variable).id()];
      }

      const F& belief(const vertex_type& vert) const{
        return beliefs_[vert.id()];
      }

      std::map<vertex_type, F> belief() const {
        std::map<vertex_type, F> ret;
        for (size_t i = 0;i < beliefs_.size(); ++i) {
          ret[factor_graph_->id2vertex(i)] = beliefs_[i];
        }
        return ret;
      }

      //! Compute the map assignment to all variables
      void map_assignment(finite_assignment &mapassg) const{
        foreach(const vertex_type &v, factor_graph_->vertices()) {
          if (v.is_variable()) {
            finite_assignment localmapassg = arg_max(beliefs_[v.id()]);
            mapassg[&(v.variable())] = localmapassg[&(v.variable())];
          }
        }
      }

      std::map<std::string, double> get_profiling_info(void) const {
        std::map<std::string, double> ret;
        ret["updates"] = numupdates_;
        ret["edgeupdates"] = numedgeupdates_;
        ret["ops"] = numops_;
        ret["runtime"] = runtime_;
        return ret;
      }
  }; // End of class parallel_wildfire_bp

}; // end of namespace

#include <sill/macros_undef.hpp>



#endif

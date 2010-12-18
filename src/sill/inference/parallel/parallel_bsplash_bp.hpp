#ifndef PARALLEL_BSPLASH_BP_HPP
#define PARALLEL_BSPLASH_BP_HPP


// STL includes
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <limits>
#include <boost/unordered_set.hpp>
// PRL Includes
#include <sill/model/factor_graph_model.hpp>
#include <sill/factor/norms.hpp>
#include <sill/math/gdl_enum.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/factor/log_table_factor.hpp>
#include <sill/parallel/binned_scheduling_queue.hpp>
#include <sill/datastructure/mutable_queue.hpp>
#include <sill/model/factor_graph_partitioning.hpp>
// Pthread tools
#include <sill/parallel/pthread_tools.hpp>


// This include should always be last
#include <sill/macros_def.hpp>

#define HARD_SPLASH_SIZE
namespace sill {

  template<typename F>
  class parallel_bsplash_bp:
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
      binned_scheduling_queue<size_t> schedule_;

      //! All threads will terminate when this is set to true
      bool finished_;
      
      //! A list of all messages. To read/write a message from src->dest
      //! you should lock the belief at the destination vertex.
      std::vector<std::map<size_t, F> > messages_;
      std::vector<double> residuals_;

      //! A list of all vertex beliefs. You shoud lock the belief before
      //! reading or writing
      std::vector<F> beliefs_;
      std::vector<spinlock> beliefslock_;
      std::vector<spinlock> vertex_being_updating_;
      
      //! the size of a splash
      size_t splash_size_;

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
      
      // profiling data
      size_t numsplashes_;
      size_t numupdates_;
      size_t numedgeupdates_;
      size_t numops_;
      double runtime_;
    public:
      /**
        Blf Splash processing thread
      */
      class blfsplash_thread : public thread {
        public:
          /// a reference to the parent class
          parallel_bsplash_bp &owner_;

          /// the index of the scheduling queue I am using
          size_t scheduleid_;
          
          /// The total number of splashes issued by this thread
          size_t numsplashes_;
          size_t numupdates_;
          size_t numedgeupdates_;
          size_t numops_;
          blfsplash_thread(parallel_bsplash_bp &owner, size_t scheduleid)
                                    :owner_(owner),scheduleid_(scheduleid) {}

          ~blfsplash_thread() {}
          
          void run() {
            std::cout << "Bsplash worker started" << std::endl;
            numsplashes_ = 0;
            numupdates_ = 0;
            numops_ = 0;
            numedgeupdates_ = 0;
            // pull a reference to the schedule I am handling
            binned_scheduling_queue<size_t>* schedule =
                                           &(owner_.schedule_);


            assert(schedule->empty() == false);
            timer ti;
            ti.start();
            double lasttime = 0.0;
            while(1) {
              std::pair<size_t, double> topvertex = schedule->deschedule_top();
              if(topvertex.second < owner_.bound_ || owner_.finished_) {
                bool converged = owner_.converged(scheduleid_);
                if (!converged) {
                  continue;
                }
                else {
                  break;
                }
              }
              size_t v = topvertex.first;


              // Execute a splash
              size_t ups, ops, edgeups;

              owner_.splash(owner_.factor_graph_->id2vertex(v),owner_.splash_size_,&ups, &ops, &edgeups);
              schedule->schedule(topvertex.first);
              // update the profiling information
              numsplashes_++;
              numupdates_+=  ups;
              numops_ +=  ops;
              numedgeupdates_ +=  edgeups;
              // display something ocassionally
              if(numsplashes_ % 128 == 0){
                if (owner_.timeout_ > 0 && ti.current_time() >= owner_.timeout_) break;
                if(scheduleid_ == 0 && ti.current_time() - lasttime > 1) {
                  std::cout << scheduleid_ << ": " << schedule->top().second << std::endl;
                  std::cout << numsplashes_ << " Splashes\n";
                  lasttime = ti.current_time();
                }
              }
            }
            std::cout << scheduleid_ << ": " << schedule->top().second << std::endl;
            std::cout << numsplashes_ << " Splashes\n";
          }
      }; // end blfsplash_thread

      /**
      * Create an engine (without allocating messages) with the factor
      * graph, splash size, convergence bound, and damping.
      */
      parallel_bsplash_bp():
          csr_(sum_product) {
        factor_graph_ = NULL;
        splash_size_ = 100;
        bound_ = 0.001;
        damping_ = 0.0;
        globalnumzero = 0;
      } // end parallel_bsplash_bp

      void clear() {
        factor_graph_ = NULL;
        splash_size_ = 100;
        bound_ = 0.001;
        damping_ = 0.0;
        schedule_.clear();

        messages_.clear();
        beliefs_.clear();
        beliefslock_.clear();
        
      }


      void initialize(factor_graph_type* factor_graph,
                      size_t splash_size,
                      size_t numprocs,
                      double bound,
                      double damping,
                      double timeout = 0.0) {
        timeout_ = timeout;
        std::vector<std::pair<size_t, double> > unused;
        schedule_.init(numprocs*2, unused);
        numprocs_ = numprocs;
        factor_graph_ = factor_graph;
        splash_size_ = splash_size;
        bound_ = bound;
        damping_ = damping;
        // Clear the messages
        messages_.clear();
        messages_.resize(factor_graph_->num_vertices());
        // Allocate all messages
        beliefs_.resize(factor_graph_->num_vertices());
        beliefslock_.resize(factor_graph_->num_vertices());
        vertex_being_updating_.resize(factor_graph_->num_vertices());
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
        schedule_.clear();
        finished_ = false;




        size_t idx = 0;

        residuals_.resize(factor_graph_->num_vertices());
        
        double initial_residual = std::numeric_limits<double>::max();
        foreach(const vertex_type& u, factor_graph_->vertices()) {
          schedule_.push(u.id(), initial_residual);
          residuals_[u.id()] = initial_residual;
          idx++;
        }

        wakeupbroadcasted_ = false;
        factor_graph_->build_work_per_update_cache();
        
      } // end of initialize


      bool alldone() {
        if (schedule_.top().second >= bound_) {
          return false;
        }
        return true;
      }

      // the splash thread will call this function
      // the schedule[scheduleid_] MUST BE LOCKED prior to entering this
      // function. When this function returns, it will still be locked.
      bool converged(size_t scheduleid) {
        while(1) {
          if (finished_) return true;
          // quickly check my own schedule and return if I am not done
          if (schedule_.top().second >= bound_) {
            return false;
          }
        // ok. looks like I am done. check everyone else's schedule
          if (alldone()) {
            std::cout << "finished" << std::endl;
            finished_ = true;
            return true;
          }
          else {
            return false;
          }
        }
      }


      double loop_to_convergence() {
        std::cout << "Started..." << std::endl;
        timer ti;
        ti.start();
        std::vector<blfsplash_thread*> threads;
        threads.resize(numprocs_);
        for (size_t i = 0;i < numprocs_; ++i) {
          threads[i] = new blfsplash_thread(*this,i);
          threads[i]->start();
        }

        numsplashes_ = 0;
        numupdates_ = 0;
        numops_ = 0;
        numedgeupdates_ = 0;
        
        for (size_t i = 0;i < numprocs_; ++i) {
          threads[i]->join();
          numsplashes_ += threads[i]->numsplashes_;
          numupdates_ += threads[i]->numupdates_;
          numops_ += threads[i]->numops_;
          numedgeupdates_ += threads[i]->numedgeupdates_;
          delete threads[i];
        }
        runtime_ = ti.current_time();
        return runtime_;
      } // end of splash_to_convergence



      // Construct a splash on a particular vertex
      void splash(const vertex_type& root, 
                  size_t splashsize, size_t *updates = NULL,
                  size_t *ops = NULL,
                  size_t *numedgeupdates = NULL) {
        size_t numupdates = 0;
        size_t numops = 0;
        size_t numedg = 0;
        std::vector<size_t> splash_order;
        // Grow a splash ordering
        generate_splash(root, splash_order, splashsize);

        // Push belief from the leaves to the root
        revforeach(const size_t vid, splash_order) {
          if (residuals_[vid] >= bound_/10.0) {
            vertex_type v = factor_graph_->id2vertex(vid);
            if (send_messages(v)) {
              numops += factor_graph_->work_per_update(vid);
              numedg += factor_graph_->num_neighbors(vid);
              numupdates++;
            }
          }
        }
        // Push belief from the root to the leaves (skipping the
        // root which was processed in the previous pass)
        foreach(const size_t vid,
                std::make_pair(++splash_order.begin(), splash_order.end())) {
          if (residuals_[vid] >= bound_ /10.0) {
            if (send_messages(factor_graph_->id2vertex(vid))) {
              numops += factor_graph_->work_per_update(vid);
              numedg += factor_graph_->num_neighbors(vid);
              numupdates++;
            }
          }
        }
        if (updates != NULL) (*updates) = numupdates;
        if (ops != NULL) (*ops) = numops;
        if (numedgeupdates != NULL) (*numedgeupdates) = numedg;
      } // End of splash_once

      size_t globalnumzero;
      /**
      * This function computes the splash ordering (a BFS) search for
      * the root vertex
      */
      void generate_splash(const vertex_type& root,
                            std::vector<size_t>& splash_order,
                            size_t splashsize) {
        // Create a set to track the vertices visited in the traversal
        size_t rootid = root.id();
        std::set<size_t> visited;
        std::list<size_t> splash_queue;
        size_t work_in_queue = 0;
        size_t work_in_splash = 0;
        // Set the root to be visited and the first element in the queue
        double limit = bound_;
        if (residuals_[rootid] >100) limit = 100;
        splash_queue.push_back(rootid);
        visited.insert(rootid);
        work_in_queue = factor_graph_->work_per_update(rootid); 
        // Grow a breath first search tree around the root
        for(; work_in_splash < splashsize
              && !splash_queue.empty();) {
          // Remove the first element
          size_t uid = splash_queue.front();
          splash_queue.pop_front();
          size_t work = factor_graph_->work_per_update(uid);
          #ifdef HARD_SPLASH_SIZE
          if (work + work_in_splash < splashsize || work_in_splash == 0) {
          #endif
            splash_order.push_back(uid);
            work_in_splash += work;
            work_in_queue -= work;
          #ifdef HARD_SPLASH_SIZE
          } else {
            work_in_queue -= work;
            continue;
          }
          #endif
          // if we still need more work for the splash
          if(work_in_queue + work_in_splash < splashsize) {
            // Insert the first element into the tree order If we need
            // more vertices then grow out more Add all the unvisited
            // neighbors to the queue
            foreach(size_t vid, factor_graph_->neighbor_ids(uid)) {
              if((residuals_[vid] >= limit)  && (visited.count(vid) == 0)) {
                if (factor_graph_->work_per_update(vid) + work_in_splash <= splashsize) {
                  splash_queue.push_back(vid);
                  work_in_queue += factor_graph_->work_per_update(vid);
                  visited.insert(vid);
                }
              }
            } // end of for each neighbors
          }
        } // End of foorloop
      } // End of Generate Splash



      // Update the message changing the destination belief and priority
      // and returning the belief residual.
      inline double update_message(const vertex_type& source,
                                  const vertex_type& target,
                                  const F& new_msg, const F& damped_msg) {

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


        
        double new_priority = residuals_[target.id()] + belief_residual;
        // For the residual to be finite
        if(!std::isfinite(new_priority)) {
          new_priority = std::numeric_limits<double>::max();
        }
        schedule_.increase_priority(target.id(), belief_residual);
        residuals_[target.id()] = new_priority;



        return belief_residual;
      } // end of update_message




      inline bool send_messages(const vertex_type& source) {
        // For each of the neighboring vertices (vertex_target) of
        // vertex compute the message from vertex to vertex_target
        if (vertex_being_updating_[factor_graph_->vertex2id(source)].try_lock() == false) return false;
        foreach(const vertex_type &dest,
                factor_graph_->neighbors(source)) {
          send_message(source, dest);
        }
        // Mark the vertex as having been visited
        residuals_[source.id()] = 0.0;
        schedule_.update(source.id(), 0.0);
        vertex_being_updating_[factor_graph_->vertex2id(source)].unlock();
        return true;
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

        //cavity.normalize();
        
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
        ret["splashes"] = numsplashes_;
        ret["updates"] = numupdates_;
        ret["edgeupdates"] = numedgeupdates_;
        ret["ops"] = numops_;
        ret["runtime"] = runtime_;
        return ret;
      }
  }; // End of class parallel_bsplash_bp
}; // end of namespace


#ifdef HARD_SPLASH_SIZE
#undef HARD_SPLASH_SIZE
#endif

#include <sill/macros_undef.hpp>



#endif




#ifndef PARALLEL_SYNCHRONOUS_BP_HPP
#define PARALLEL_SYNCHRONOUS_BP_HPP


// STL includes
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <limits>

// SILL Includes
#include <sill/inference/interfaces.hpp>
#include <sill/model/factor_graph_model.hpp>
#include <sill/factor/util/norms.hpp>
#include <sill/math/gdl_enum.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/factor/canonical_table.hpp>
#include <sill/datastructure/mutable_queue.hpp>
#include <sill/model/factor_graph_partitioning.hpp>
#include <sill/datastructure/circular_queue.hpp>
// Pthread tools
#include <sill/parallel/pthread_tools.hpp>


// This include should always be last
#include <sill/macros_def.hpp>

namespace sill {

  template<typename F>
  class parallel_synchronous_bp:
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

      //! The lock for each queue
      std::vector<spinlock> scheduleslock_;

      //! All threads will terminate when this is set to true
      bool finished_;
      
      //! A list of all messages. To read/write a message from src->dest
      //! you should lock the belief at the destination vertex.
      std::vector<std::map<size_t, F> > messages_;
      std::vector<std::map<size_t, F> > lastmessages_;

      std::vector<double> residuals_;
      //! A list of all vertex beliefs. 
      std::vector<F> beliefs_;
      std::vector<F> lastbeliefs_;
      //! Number of processors
      size_t numprocs_;
      
      //! convergence bound
      double bound_;

      //! level of damping 1.0 is fully damping and 0.0 is no damping
      double damping_;
      double timeout_;

      //! the commutative semiring for updates (typically sum_product)
      commutative_semiring csr_;

      size_t numupdates_;
      size_t numops_;
      size_t numedgeupdates_;
      double runtime_;
      size_t numiterations_;
    public:
      /**
        Blf Splash processing thread
      */
      class synchronous_thread : public thread {
        public:
          /// a reference to the parent class
          parallel_synchronous_bp &owner_;
          
          std::set<size_t> update_set_;
          
          double max_change_;

          barrier &barrier_;
          
          synchronous_thread(parallel_synchronous_bp &owner, 
                             std::set<size_t> &update_set,
                             barrier &syncbarrier)  
                                    :owner_(owner),
                                    update_set_(update_set),
                                    barrier_(syncbarrier) {}

          ~synchronous_thread() {}
          
          void run() {
            std::cout << "Starting Run in worker" << std::endl;
            
            // while we are not done
            while(owner_.finished() == false) {
            
              // I need 3 calls to barrier! 
              // stage 1: Update messages
              // send all the messages in my updateset, and update the maxchange
              max_change_ = 0;
              foreach(size_t vid, update_set_) {
                owner_.send_messages(owner_.factor_graph_->id2vertex(vid));
              }
              foreach(size_t vid, update_set_) {
                max_change_ = std::max(max_change_, owner_.residuals_[vid]);
              }
              barrier_.wait();

              // stage 2: Update beliefs
              foreach(size_t vid, update_set_) {
                owner_.update_belief(owner_.factor_graph_->id2vertex(vid));
                owner_.residuals_[vid] = 0.0;
              }
              barrier_.wait();
              
              // stage 3: Synchronize with mainthread and swap pointers
              barrier_.wait();
            }
          }
      }; // end synchronous_thread

      /**
      * Create an engine (without allocating messages) with the factor
      * graph, splash size, convergence bound, and damping.
      */
      parallel_synchronous_bp():
          csr_(sum_product) {
        factor_graph_ = NULL;
        bound_ = 0.001;
        damping_ = 0.0;
      } // end parallel_msplash_bp

      void clear() {
        factor_graph_ = NULL;
        bound_ = 0.001;
        damping_ = 0.0;

        messages_.clear();
        lastmessages_.clear();
        beliefs_.clear();
        lastbeliefs_.clear();
        finished_ = false;
      }

      bool finished() {
        return finished_;
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
        residuals_.clear();
        residuals_.resize(factor_graph_->num_vertices());
        messages_.resize(factor_graph_->num_vertices());
        beliefs_.resize(factor_graph_->num_vertices());
        // Allocate all messages
        foreach(const vertex_type& u, factor_graph_->vertices()) {
          foreach(const size_t vid, factor_graph_->neighbor_ids(u)) {
            vertex_type v = factor_graph_->id2vertex(vid);

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
            blf.normalize();
          } else {
            blf = F(make_domain(&(u.variable())),1.0).normalize();
          }
        }
        lastmessages_ = messages_;
        lastbeliefs_ = beliefs_;
        factor_graph_->build_work_per_update_cache();
      } // end of initialize



      double loop_to_convergence() {
        finished_ = false;
        numiterations_ = 0;
        barrier syncbarrier(numprocs_ + 1);
        std::cout << "Started..." << std::endl;
        
        std::vector<synchronous_thread*> threads;
        threads.resize(numprocs_);

        // copy all the vertices into a vector
        std::vector<size_t> allvertices;

        for (size_t i = 0;i < factor_graph_->num_vertices(); ++i) {
          allvertices.push_back(i);
        }

        size_t totalcostperiteration = 0;
        size_t numedgesperiteration = 0;
        foreach(size_t vid, allvertices) {
          totalcostperiteration += factor_graph_->work_per_update(vid);
          numedgesperiteration += factor_graph_->num_neighbors(vid);
        }
        // size of each thread's chunk
        size_t chunksize = (allvertices.size() / numprocs_);
        
        // loop over each thread and allocate a chunk
        for (size_t i = 0;i < numprocs_; ++i) {
          size_t startv = i * chunksize;
          size_t endv = (i+1) * chunksize - 1;
          if (i == numprocs_ - 1) endv = allvertices.size() - 1;

          // create the chunk
          std::set<size_t> jobchunk;
          std::copy(allvertices.begin() + startv, 
                    allvertices.begin() + endv + 1,
                    std::inserter(jobchunk,jobchunk.end()));
          // create the thread
          threads[i] = new synchronous_thread(*this,jobchunk,syncbarrier);
        }
        timer ti;
        ti.start();

        // start all threads!
        for (size_t i = 0;i < numprocs_; ++i) {
          threads[i]->start();
        }
        numupdates_ = 0;
        numops_ = 0;
        numedgeupdates_ = 0;
        numiterations_ = 0;
        // loop while the largest message change is still large
        while (1) {
          syncbarrier.wait();
          syncbarrier.wait();
          /*for (size_t i = 0; i < beliefs_.size(); ++i) {
            std::cout << beliefs_[i];
            std::cout << "\n";
          }
          std::cout << "\n-----------------------\n"; */
          //lastbeliefs_ = beliefs_;
          //lastmessages_ = messages_;
          lastmessages_.swap(messages_);
          ++numiterations_;
          
          double maxchange = 0;
          for (size_t i = 0;i < numprocs_; ++i) {
            maxchange = std::max(maxchange, threads[i]->max_change_);
          }
          std::cout << maxchange << std::endl;
          if (maxchange < bound_ || (timeout_ > 0 && ti.current_time() >= timeout_) ) {
            finished_ = true;
            break;
          }
          syncbarrier.wait();
        }
        
        syncbarrier.wait();

        numupdates_ = numiterations_ * allvertices.size();
        numops_ = numiterations_ * totalcostperiteration;
        numedgeupdates_ = numiterations_ * numedgesperiteration;
        
        for (size_t i = 0;i < numprocs_; ++i) {
          threads[i]->join();
          delete threads[i];
        }

        runtime_ = ti.current_time();
        return runtime_;
      } // end of splash_to_convergence



  
      inline void update_message(const vertex_type& source,
                                const vertex_type& target,
                                const F& new_msg,
                                 const F& damped_msg) {
        F& original_msg = lastmessages_[source.id()][target.id()];
        F blf = lastbeliefs_[target.id()];
  
        // Update the new belief by dividing out the old message and
        // multiplying in the new message.
        blf.combine_in(original_msg, divides_op);
        blf.combine_in(new_msg, csr_.dot_op);
        blf.normalize();
        
        double residual = norm_1(blf,lastbeliefs_[target.id()]);

        blf.combine_in(new_msg, divides_op);
        blf.combine_in(damped_msg, csr_.dot_op);
        blf.normalize();
        // Get the original message
        residuals_[target.id()] += residual;

        messages_[source.id()][target.id()] = damped_msg;
        
      } // end of update_message
  
  
      void update_belief(const vertex_type& v) {
        F &blf = beliefs_[v.id()];
        if(v.is_factor()) {
          blf = v.factor();
        } else {
          blf = F(make_domain(&(v.variable())),1.0).normalize();
        }
  
        foreach(const size_t sourceid,
                factor_graph_->neighbor_ids(v)) {
          blf.combine_in(messages_[sourceid][v.id()], csr_.dot_op);
        }
        blf.normalize();
        lastbeliefs_[v.id()] = blf;
        //lastmessages_[v.id()] = messages_[v.id()];
      }
  
      inline void send_messages(const vertex_type& source) {
        // For each of the neighboring vertices (vertex_target) of
        // vertex compute the message from vertex to vertex_target
        foreach(const vertex_type& dest,
                factor_graph_->neighbors(source)) {
          send_message(source, dest);
        }
      } // end of update messages
  
  
  
      inline void send_message(const vertex_type& source,
                              const vertex_type& target) {
        // read the last messages and beliefs and write it to the new messages
        // and new beliefs
        F& blf = lastbeliefs_[source.id()];
  
        // Construct the cavity
        F cavity = combine(blf, lastmessages_[target.id()][source.id()], divides_op);
        // Marginalize out any other variables
        typename F::domain_type domain = make_domain(source.is_variable()?
                                                      &(source.variable()) :
                                                      &(target.variable()));
  
        F new_msg = cavity.collapse(csr_.cross_op, domain);
        // Normalize the message
        new_msg.normalize();
        F damped_msg;
        // Damp messages form factors to variables
        if(target.is_variable()) {
          damped_msg = weighted_update(new_msg,
                                    lastmessages_[source.id()][target.id()],
                                    damping_);
        }
        else {
          damped_msg = new_msg;
        }
        // update the message also updating the schedule
        update_message(source, target, new_msg, damped_msg);
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
      const F& belief(variable_type variable) const{
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
        ret["ops"] = numops_;
        ret["runtime"] = runtime_;
        ret["iterations"] = numiterations_;
        ret["edgeupdates"] = numedgeupdates_;
        return ret;
      }
    }; // End of class parallel_synchronous_bp

}; // end of namespace

#include <sill/macros_undef.hpp>



#endif

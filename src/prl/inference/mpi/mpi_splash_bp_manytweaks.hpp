#ifndef MPI_SPLASH_BP_HPP
#define MPI_SPLASH_BP_HPP


// STL includes
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>

// PRL Includes
#include <prl/base/stl_util.hpp>
#include <prl/inference/interfaces.hpp>
#include <prl/mpi/mpi_wrapper.hpp>
#include <prl/mpi/mpi_consensus.hpp>
#include <prl/inference/mpi/mpi_adapter.hpp>
#include <prl/model/factor_graph_model.hpp>
#include <prl/factor/norms.hpp>
#include <prl/math/gdl_enum.hpp>
#include <prl/factor/table_factor.hpp>
#include <prl/datastructure/mutable_queue.hpp>
#include <prl/model/factor_graph_partitioning.hpp>

// This include should always be last
#include <prl/macros_def.hpp>

//#define HARD_SPLASH_SIZE
namespace prl {

  template<typename F>
  class mpi_splash_bp:public factor_graph_inference<factor_graph_model<F> > {
    ///////////////////////////////////////////////////////////////////////
    // typedefs
  public:
    typedef F factor_type;
    typedef factor_graph_model<F> model_type;
    typedef typename factor_graph_model<F>::variable_type    variable_type;
    typedef typename factor_graph_model<F>::vertex_type      vertex_type;

    //! IMPORTANT map is in [dest][src] form
    typedef std::map<vertex_type,
                     std::map<vertex_type, F> > message_map_type;

    typedef std::map<vertex_type, F> belief_map_type;
    ///////////////////////////////////////////////////////////////////////
    // Data members
  private:

   
    mpi_post_office &po_;
    mpi_consensus *consensus_;
    mpi_adapter< mpi_splash_bp<F> > *adapter_;
    
    //! true if I am the root nose
    bool isroot_;
    bool blockremotesends_;

    //! pointer to the universe containing all the variables in the factor graph
    universe *universe_;

    //! The root node's pointer to the complete factor graph
    factor_graph_model<F>* root_factor_graph_;

    //! pointer to the local subset of the factor graph
    factor_graph_model<F>* local_graph_;
  
    //! the schedule of vertices
    mutable_queue<size_t, double> schedule_;

    std::map<size_t, double> boundary_true_residual_;


    //! messages
    std::vector<std::map<size_t, F> >  messages_;

    //! beliefs
    std::vector<F> beliefs_;

    std::vector<F> rootbeliefs_;

    //! the size of a splash
    size_t splash_size_;

    //! convergence bound
    double bound_;

    //! level of damping 1.0 is fully damping and 0.0 is no damping
    double damping_;

    //! the commutative semiring for updates (typically sum_product)
    commutative_semiring csr_;

    uint64_t numsplashes_;
    uint64_t boundaryupdates_;
    uint64_t interiorupdates_;
    uint64_t numupdates_;
    uint64_t numedgeupdates_;
    uint64_t numops_;
    uint64_t mpimessagessent_;
    double runtime_;

    size_t timeout_;
  public:
    

    mpi_splash_bp(mpi_post_office &po): po_(po), csr_(sum_product) {
      universe_ = NULL;
      local_graph_ = NULL;
      root_factor_graph_ = NULL;
      splash_size_ = 0;
      bound_ = 1E-5;
      damping_ = 0.1;
      consensus_ = NULL;
      adapter_ = NULL;
    }
    
    /**
      * initializes the engine with an instance of a factor graph
      * and the corresponding universe.
      * This is called by the root node. the other nodes call wait()
      * This function will partition the factor graph and transmit it to
      * the other nodes.
      */
    void initialize_root(universe* universe,
                    factor_graph_model<F>* factor_graph,
                    size_t splash_size,
                    double bound,
                    double damping,
                    typename factor_graph_partition<F>::algorithm partitionalgorithm =
                        factor_graph_partition<F>::KMETIS,
                    size_t overpartitionfactor = 1, 
                    size_t timeout = 0) {
      // copy the parameters into the internal fields
      universe_ = universe;
      root_factor_graph_ = factor_graph;
      rootbeliefs_.resize(root_factor_graph_->num_vertices());
      splash_size_ = splash_size;
      bound_ = bound;
      damping_ = damping;
      isroot_ = true;
      timeout_ = timeout;
      consensus_ = new mpi_consensus(po_);
      adapter_ = new mpi_adapter< mpi_splash_bp<F> >(po_, *consensus_);
      po_.barrier();
      // partition the graph
      factor_graph_partition<F> partition(*root_factor_graph_,
                                    overpartitionfactor * po_.num_processes(),
                                    partitionalgorithm,
                                    false);

      adapter_->initialize_root(*universe_,
                                *root_factor_graph_,
                                partition);
      po_.barrier();
      // transmit the parameters
      {
        std::string str;
        std::stringstream strm(str);
        oarchive arc(strm);
        arc << splash_size_ << bound_ << damping_;
        adapter_->sync_send_all_but_self(strm.str());
      }
      
      local_graph_ = &(adapter_->get_local_graph());
      create_messages();
      std::cout << po_.id() << " initialized\n";
      std::cout.flush();
      
      po_.barrier();
      
    }

    void initialize_nonroot(size_t timeout = 0) {
      isroot_ = false;
      universe_ = new universe;
      consensus_ = new mpi_consensus(po_);
      adapter_ = new mpi_adapter< mpi_splash_bp<F> >(po_, *consensus_);
      timeout_ = timeout;
      po_.barrier();
      adapter_->initialize_nonroot(*universe_);
      // receive the parameters
      po_.barrier();
      {
        std::string str = adapter_->sync_recv();
        std::stringstream strm(str);
        iarchive arc(strm);
        arc >> splash_size_ >> bound_ >> damping_;
        
      }
      local_graph_ = &(adapter_->get_local_graph());
      create_messages();
      std::cout << po_.id() << " initialized\n";
      std::cout.flush();
      
      po_.barrier();
      //consensus_ = new mpi_consensus(po);
    }
    void clear() {
      delete consensus_;
      delete adapter_;
      if (isroot_ == false) delete universe_;
    
      local_graph_ = NULL;
      consensus_ = NULL;
      adapter_ = NULL;
      universe_ = NULL;
      root_factor_graph_ = NULL;
      beliefs_.clear();
      rootbeliefs_.clear();
      messages_.clear();
      schedule_.clear();
    }

    
    void create_messages() {
      // Clear the messages
      messages_.clear();
      messages_.resize(local_graph_->num_vertices());
      beliefs_.resize(local_graph_->num_vertices());

      // Allocate all messages
      // loop through all the vertices are are truly mine
      foreach(const vertex_type& src, local_graph_->vertices()) {
        if (adapter_->own_vertex(src)) {
          // construct a message to all neighbors
          foreach(const vertex_type& dest, local_graph_->neighbors(src)) {
            // note that messages_[src][dest] will also allocate the messages
            // if it has not been already allocated
            F& msg = messages_[src.id()][dest.id()];
            typename F::domain_type domain = make_domain(dest.is_variable() ?
                                                        &(dest.variable()) :
                                                        &(src.variable()));
            msg = F(domain, 1.0).normalize();

            // construct messages in both directions.
            F& msg2 = messages_[dest.id()][src.id()];
            domain = make_domain(src.is_variable() ?
                                &(src.variable()) :
                                &(dest.variable()));
            msg2 = F(domain, 1.0).normalize();
          }
          // Initialize the belief. also observe that 
          // beliefs_[src] will allocate the beliefs
          F& blf = beliefs_[src.id()];
          if(src.is_factor()) {
            blf = src.factor();
          } else {
            blf = F(make_domain(&(src.variable())),1.0).normalize();
          }
        }
      }
      // Initialize the schedule
      schedule_.clear();
      double initial_residual = std::numeric_limits<double>::max();
      foreach(const vertex_type& v, local_graph_->vertices()) {
        if (adapter_->own_vertex(v)) {
          schedule_.push(v.id(), initial_residual);
          if (adapter_->on_boundary(v)) {
            boundary_true_residual_[v.id()] = 0;
          }
        }
      }
    }

    void root_update_belief(vertex_type v, F &b) {
      rootbeliefs_[v.id()] = b;
    }
    void update_message_from_remote(vertex_type src, vertex_type dest, F &msg) {
      assert(adapter_->own_vertex(dest));
      assert(!adapter_->own_vertex(src));
      update_message(src, dest,msg, false);
    }

    inline void update_message(const vertex_type& source,
                               const vertex_type& target,
                               const F& new_msg, bool sendremotes) {
      size_t srcid = source.id();
      size_t targetid = target.id();
      // if the destination is not local just record and return
      if(adapter_->own_vertex(target) == false) {
        double delta = norm_1(messages_[srcid][targetid], new_msg);
        messages_[srcid][targetid] = new_msg;
        if (sendremotes) {
  	      adapter_->queue_outgoing_bp_message(source, target, new_msg, delta);
        }
        return;
      }
      assert(messages_.size() > srcid);
      assert(messages_[srcid].find(targetid) != messages_[srcid].end());
      // Get the original message
      F& original_msg = messages_[srcid][targetid];
      
      F prevblf = beliefs_[targetid];
      F& blf = beliefs_[targetid];
      blf.combine_in(original_msg, divides_op);
      blf.combine_in(new_msg, csr_.dot_op);
      blf.normalize();
      //      assert(blf.minimum() > 0.0);
      original_msg = new_msg;


      double new_residual = schedule_.get(targetid);

      double delta = norm_1(prevblf,blf);
      new_residual += delta;

      // not sure if this is necessary.
      if (std::isnan(new_residual))
        new_residual = std::numeric_limits<double>::infinity();

      // Update the residual
      //if(schedule_.get(target) < new_residual )
      
      // boundary_true_residual tracks the actual residual value
      if (adapter_->on_boundary(target)) {
        boundary_true_residual_[targetid] += delta;
        if (boundary_true_residual_[targetid] > bound_) {
          new_residual = boundary_true_residual_[targetid];
        }
      }
      schedule_.update(targetid, new_residual);
    } // end of update_message

    inline void send_messages(const vertex_type& source) {

      assert(adapter_->own_vertex(source));
      std::cout.flush();
      //updates_out_ << factor_graph_->vertex2id(source) << std::endl;
      // For each of the neighboring vertices (vertex_target) of
      // vertex compute the message from vertex to vertex_target
      bool isremote = false;
      bool sendremotes = false;
      if (schedule_.get(source.id()) > bound_) {
        sendremotes = true;
      }
      foreach(const vertex_type& dest,
              local_graph_->neighbors(source)) {
        if (adapter_->own_vertex(dest) == false) isremote = true;
        send_message(source, dest, sendremotes);
      }
      if (isremote) {
        boundaryupdates_++;
      }
      else {
        interiorupdates_++;
      }
      // Mark the vertex as having been visited
      if (isremote == true && sendremotes) {
          boundary_true_residual_[source.id()] = 0.0;
      }
      schedule_.update(source.id(), 0.0);
      // update the belief
    } // end of update messages



    inline void send_message(const vertex_type& source,
                             const vertex_type& target,
                             bool sendremotes) {
      // here we can assume that the current belief is correct
      F& blf = beliefs_[source.id()];

      // Construct the cavity
      F cavity = combine(blf, messages_[target.id()][source.id()], divides_op);
      // Marginalize out any other variables
      typename F::domain_type domain = make_domain(source.is_variable()?
                                                    &(source.variable()) :
                                                    &(target.variable()));

      F new_msg = cavity.collapse(csr_.cross_op, domain);
      // Normalize the message
      new_msg.normalize();
      // Damp messages form factors to variables
      if(target.is_variable()) {
        new_msg = weighted_update(new_msg,
                                  messages_[source.id()][target.id()],
                                  damping_);
      }
      // update the message also updating the schedule
      update_message(source, target, new_msg,sendremotes);
    } // end of send_message

    void splash(const vertex_type& root) {
      std::vector<size_t> splash_order;
      // Grow a splash ordering
      generate_splash(root, splash_order);
      // Push belief from the leaves to the root
      revforeach(size_t vid, splash_order) {
        if (schedule_.get(vid) >= bound_ / 10.0) {
          send_messages(local_graph_->id2vertex(vid));
        }
      }
      // Push belief from the root to the leaves (skipping the
      // root which was processed in the previous pass)
      for (size_t i = 1; i < splash_order.size(); ++i) {
        if (schedule_.get(splash_order[i]) >= bound_ / 10.0) {
          send_messages(local_graph_->id2vertex(splash_order[i]));
        }
      }
    } // End of splash_once


      /**
      * This function computes the splash ordering (a BFS) search for
      * the root vertex
      */

      void generate_splash(const vertex_type& root,
                            std::vector<size_t>& splash_order) {

        // Create a set to track the vertices visited in the traversal
        std::set<size_t> visited;
        std::list<size_t> splash_queue;
        size_t work_in_queue = 0;
        size_t work_in_splash = 0;

        // Set the root to be visited and the first element in the queue
        splash_queue.push_back(root.id());
        work_in_queue += local_graph_->work_per_update(root);
        visited.insert(root.id());
        
        numops_ += local_graph_->work_per_update(root);
        numedgeupdates_ += local_graph_->num_neighbors(root);
        // Grow a breath first search tree around the root
        for(work_in_splash = 0; work_in_splash < splash_size_
              && !splash_queue.empty();) {
          // Remove the first element
          size_t uid = splash_queue.front();
          splash_queue.pop_front();
          size_t work = local_graph_->work_per_update(uid);
//          if(adapter_->on_boundary(u) == false) work = work*2;
          #ifdef HARD_SPLASH_SIZE
          if (work + work_in_splash < splash_size_ || work_in_splash == 0) {
          #endif
            splash_order.push_back(uid);
            work_in_splash += work;
            work_in_queue -= work;
            numops_ += 2 * local_graph_->work_per_update(uid);
            numedgeupdates_ += 2 * local_graph_->num_neighbors(uid);
          #ifdef HARD_SPLASH_SIZE
          } else {
            work_in_queue -= work;
            continue;
          }
          #endif
          // if we still need more work for the splash
          if(work_in_queue + work_in_splash < splash_size_) {
            // Insert the first element into the tree order If we need
            // more vertices then grow out more Add all the unvisited
            // neighbors to the queue
            foreach(const vertex_type& v, local_graph_->neighbors(uid)) {
              if((adapter_->own_vertex(v)) && (visited.count(v.id()) == 0)
                && schedule_.get(v.id()) > bound_/10.0) {
                splash_queue.push_back(v.id());
                visited.insert(v.id());
                work_in_queue += local_graph_->work_per_update(v);
              }
            } // end of for each neighbors
          }
        } // End of foorloop
        numsplashes_++;
        numupdates_ += 2 * splash_order.size() - 1;
        std::cout.flush();
        
      } // End of Generate Splash

      
      // returns number of seconds spent in splash
      double loop_to_convergence() {
        std::cout << "schedule size " << schedule_.size() << "\n";
        std::cout.flush();
        std::cout << "splash to convergence...\n";
        std::cout.flush();
        
        std::ofstream fout;
        std::stringstream ss;
        ss << "log_" << po_.id();
        fout.open(ss.str().c_str());
        
        numsplashes_ = 0;
        boundaryupdates_ = 0;
        interiorupdates_ = 0;
        numupdates_ = 0;
        numedgeupdates_ = 0;
        numops_ = 0;
        runtime_ = 0.0;
        size_t nump = po_.num_processes();
        po_.barrier();
        timer ti;
        ti.start();
        fout << "started " << ti.current_time() << std::endl;
        double lasttime = ti.current_time();
        //assert(schedule_.size() > 0);
        size_t update_count = 0;
        while(1) {
          bool justwokeup = true;
          while(1) {
            // Receive any inbound messages
            adapter_->receive_bp_messages(this);

            consensus_->begin_critical_section();
            double curtop = schedule_.top().second; 
            if (justwokeup || nump == 1) {
              if (timeout_ > 0 && ti.current_time() > timeout_ ||
                  schedule_.size() == 0 || 
                  curtop <= bound_) {
                break;
              }
            }
            else {
              if (timeout_ > 0 && ti.current_time() > timeout_ ||
                  schedule_.size() == 0 || 
                  curtop <= bound_  / 5.0) {
                break;
              }
            }
            consensus_->end_critical_section();
            justwokeup = false;
            splash(local_graph_->id2vertex(schedule_.top().first));
            update_count++;
            if((ti.current_time() - lasttime) >= 4) {
              lasttime = ti.current_time();
              std::cout << schedule_.top().second << ": "
              << double(boundaryupdates_)/(interiorupdates_+boundaryupdates_)
              << " " << numsplashes_ << " "
              << double(numupdates_)/numsplashes_ << std::endl;
            }
           
            // Send newly created outbound messages
//            if (curtop > bound_) {
              adapter_->flush();
//            }
          }  // While great than bound
          
          // Send any remaining messages
          adapter_->flush();

//          std::cout << "stopped " << ti.current_time() << std::endl;
          if (consensus_->end_critical_section_and_finish()) break;
//          std::cout << "started " << ti.current_time() << std::endl;
          adapter_->receive_bp_messages(this);
        } // end of while(finished)
//        std::cout<< "stopped " << ti.current_time() << std::endl;
        fout.close();
        std::cout << "Finished Splash to Convergence" << std::endl;
        
        runtime_ = ti.current_time();
        if (isroot_) {
          std::cout << "receiving beliefs...\n";
          adapter_->send_beliefs_to_root(beliefs_);
          adapter_->root_receive_beliefs(this);
        }
        else {
          std::cout << "sending beliefs...\n";
          adapter_->send_beliefs_to_root(beliefs_);
        }
        std::cout << "done\n";
        po_.barrier();
        synchronize_stats();
        return runtime_;
      } // End of run


      const F& belief(variable_type* variable) const {
        return belief(root_factor_graph_->to_vertex(variable));
      } // end of send_message

      const F& belief(const vertex_type& vert) const {
        assert(isroot_);
        return rootbeliefs_[vert.id()];
      } // end of send_message


      std::map<vertex_type, F> belief() const{
        assert(isroot_);
        std::map<vertex_type, F> ret;
        for (size_t i = 0;i < rootbeliefs_.size(); ++i) {
          ret[root_factor_graph_->id2vertex(i)] = rootbeliefs_[i];
        }
        return ret;
      } // end of send_message

      void map_assignment(finite_assignment &mapassg) const {
        assert(isroot_);
        foreach(const vertex_type &v, root_factor_graph_->vertices()) {
          if (v.is_variable()) {
            finite_assignment localmapassg = arg_max(belief(v));
            mapassg[&(v.variable())] = localmapassg[&(v.variable())];
          }
        }
      }

      void synchronize_stats() {
        std::cout << "synchronizing stats..." << std::endl;
        mpimessagessent_ = adapter_->nummessagessent();
        mpi_inplace_reduce_uint64(mpimessagessent_);
        mpi_inplace_reduce_uint64(numsplashes_);
        mpi_inplace_reduce_uint64(numupdates_);
        mpi_inplace_reduce_uint64(numedgeupdates_);
        mpi_inplace_reduce_uint64(boundaryupdates_);
        mpi_inplace_reduce_uint64(interiorupdates_);
        mpi_inplace_reduce_uint64(numops_);
      }

      std::map<std::string, double> get_profiling_info(void) const {
        std::map<std::string, double> ret;
        ret["splashes"] = numsplashes_;
        ret["updates"] = numupdates_;
        ret["boundaryupdates"] = boundaryupdates_;
        ret["interiorupdates"] = interiorupdates_;
        ret["edgeupdates"] = numedgeupdates_;
        ret["ops"] = numops_;
        ret["runtime"] = runtime_;
        ret["nummpimessages"] = mpimessagessent_;
        return ret;
      }
  }; // End of class mpi_splash_bp




}; // end of namespace

#ifdef HARD_SPLASH_SIZE
#undef HARD_SPLASH_SIZE
#endif
#include <prl/macros_undef.hpp>



#endif // MPI_SPLASH_BP_HPP

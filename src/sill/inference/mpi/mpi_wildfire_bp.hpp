#ifndef MPI_WILDFIRE_BP
#define MPI_WILDFIRE_BP


// STL includes
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>

// PRL Includes
#include <sill/inference/interfaces.hpp>
#include <sill/mpi/mpi_wrapper.hpp>
#include <sill/inference/mpi/mpi_adapter.hpp>
#include <sill/model/factor_graph_model.hpp>
#include <sill/factor/util/norms.hpp>
#include <sill/math/gdl_enum.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/datastructure/mutable_queue.hpp>
#include <sill/model/factor_graph_partitioning.hpp>

// This include should always be last
#include <sill/macros_def.hpp>

// if this is on, each machine will loop independently, and use distributed
// consensus in the same way as the MPI_splash_engine
#define ASYNCHRONOUS_TERMINATION

namespace sill {

  template<typename F>
  class mpi_wildfire_bp:
            public factor_graph_inference<factor_graph_model<F> > {
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
    mpi_adapter< mpi_wildfire_bp<F> > *adapter_;
    
    //! true if I am the root nose
    bool isroot_;
    
    //! pointer to the universe containing all the variables in the factor graph
    universe *universe_;

    //! The root node's pointer to the complete factor graph
    factor_graph_model<F>* root_factor_graph_;

    //! pointer to the local subset of the factor graph
    factor_graph_model<F>* local_graph_;
  
    //! the schedule of vertices
    mutable_queue<size_t, double> schedule_;

    //! messages
    std::vector<std::map<size_t, F> >  messages_;

    //! beliefs
    std::vector<F> beliefs_;

    std::vector<F> rootbeliefs_;

    //! convergence bound
    double bound_;

    //! level of damping 1.0 is fully damping and 0.0 is no damping
    double damping_;

    //! the commutative semiring for updates (typically sum_product)
    commutative_semiring csr_;

    uint64_t boundaryupdates_;
    uint64_t boundaryedgeupdates_;
    uint64_t interiorupdates_;
    uint64_t numupdates_;
    uint64_t numedgeupdates_;
    uint64_t numops_;
    uint64_t mpimessagessent_;
    uint64_t mpibytessent_;
    double runtime_;
    size_t timeout_;
  public:

    mpi_wildfire_bp(mpi_post_office &po): po_(po), csr_(sum_product) {
      universe_ = NULL;
      local_graph_ = NULL;
      root_factor_graph_ = NULL;
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
                    double bound,
                    double damping,
                    typename factor_graph_partition<F>::algorithm partitionalgorithm =
                        factor_graph_partition<F>::KMETIS,
                    size_t overpartitionfactor = 1, size_t timeout = 0) {
      // copy the parameters into the internal fields
      universe_ = universe;
      root_factor_graph_ = factor_graph;
      rootbeliefs_.resize(root_factor_graph_->num_vertices());
      bound_ = bound;
      damping_ = damping;
      isroot_ = true;
      timeout_ = timeout;
      consensus_ = new mpi_consensus(po_);
      adapter_ = new mpi_adapter< mpi_wildfire_bp<F> >(po_, *consensus_);
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
        arc << bound_ << damping_;
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
      adapter_ = new mpi_adapter< mpi_wildfire_bp<F> >(po_, *consensus_);
      timeout_ = timeout;
      po_.barrier();
      adapter_->initialize_nonroot(*universe_);
      // receive the parameters
      po_.barrier();
      {
        std::string str = adapter_->sync_recv();
        std::stringstream strm(str);
        iarchive arc(strm);
        arc >> bound_ >> damping_;
        
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
        }
      }

    }

    void root_update_belief(vertex_type v, F &b) {
      rootbeliefs_[v.id()] = b;
    }
    void update_message_from_remote(vertex_type src, vertex_type dest, F &msg) {
      assert(adapter_->own_vertex(dest));
      assert(!adapter_->own_vertex(src));
      update_message(src, dest,msg);
    }

    inline void update_message(const vertex_type& source,
                               const vertex_type& target,
                               const F& new_msg) {
      // if the destination is not local just record and return
      size_t srcid = source.id();
      size_t targetid = target.id();

      if(adapter_->own_vertex(target) == false) {
        double delta = norm_1(messages_[srcid][targetid], new_msg);
        messages_[srcid][targetid] = new_msg;
        adapter_->queue_outgoing_bp_message(source, target, new_msg, delta);
        boundaryedgeupdates_++;
        return;
      }
      assert(messages_.size() > srcid);
      assert(messages_[srcid].find(targetid) != messages_[srcid].end());

      F damped_msg = new_msg;
      if(target.is_variable()) {
        damped_msg = weighted_update(new_msg,
                                  messages_[srcid][targetid],
                                  damping_);
      }
 
      // Get the original message
      F& original_msg = messages_[srcid][targetid];

      F prevblf = beliefs_[targetid];
      F& blf = beliefs_[targetid];
      blf.combine_in(original_msg, divides_op);
      blf.combine_in(new_msg, csr_.dot_op);
      blf.normalize();
      double delta=norm_1(prevblf, blf); 
      //      assert(blf.minimum() > 0.0);
      blf.combine_in(new_msg, divides_op);
      blf.combine_in(damped_msg, csr_.dot_op);
      blf.normalize();

      //      assert(blf.minimum() > 0.0);
      original_msg = damped_msg;

      double new_residual = schedule_.get(targetid);

      new_residual+=delta;

      // not sure if this is necessary.
      if (std::isnan(new_residual))
        new_residual = std::numeric_limits<double>::infinity();

      // Update the residual
      //if(schedule_.get(target) < new_residual )
      schedule_.update(targetid, new_residual);
    } // end of update_message

    inline void send_messages(const vertex_type& source) {

      numupdates_++;
      numedgeupdates_ += local_graph_->num_neighbors(source);
      numops_ += local_graph_->work_per_update(source);;

      assert(adapter_->own_vertex(source));
      std::cout.flush();
      //updates_out_ << factor_graph_->vertex2id(source) << std::endl;
      // For each of the neighboring vertices (vertex_target) of
      // vertex compute the message from vertex to vertex_target
      bool isremote = false;
      foreach(const vertex_type& dest,
              local_graph_->neighbors(source)) {
        if (adapter_->own_vertex(dest) == false) isremote = true;
        send_message(source, dest);
      }

      if (isremote) {
        boundaryupdates_++;
      }
      else {
        interiorupdates_++;
      }

      // Mark the vertex as having been visited
      schedule_.update(source.id(), 0.0);
      // update the belief
    } // end of update messages



    inline void send_message(const vertex_type& source,
                             const vertex_type& target) {
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
      // update the message also updating the schedule
      update_message(source, target, new_msg);
    } // end of send_message

      
      // returns number of seconds spent in splash
      double loop_to_convergence() {
        adapter_->clearstats();

        std::cout << "loop to convergence...\n";
        std::cout.flush();

        boundaryupdates_ = 0;
        boundaryedgeupdates_ = 0;
        interiorupdates_ = 0;
        numupdates_ = 0;
        numedgeupdates_ = 0;
        numops_ = 0;
        runtime_ = 0.0;

        // create log files
        std::ofstream fout;
        std::stringstream ss;
        ss << "log_" << po_.id();
        fout.open(ss.str().c_str());

        
        // synchronouze and start the timer
        po_.barrier();
        timer ti;
        ti.start();
        fout << "started " << ti.current_time() << std::endl;
        double lasttime = ti.current_time();
        size_t iterations = 0;
        bool outoftime = false;
        // loop to convergence
        while(1) {
          if (outoftime == false) {
            iterations++;
            foreach(const vertex_type& v, local_graph_->vertices()) {
// if asynchronous termination is on, I must avoid sending spurious messages
#ifdef ASYNCHRONOUS_TERMINATION
              if (schedule_.top().second <= bound_) break;
#endif
              if(adapter_->own_vertex(v) && schedule_[v.id()] > bound_) {
                send_messages(v);
              }
              adapter_->flush(this, false);
            }
          }
          adapter_->flush(this);
          if((ti.current_time() - lasttime) >= 2) {
            lasttime = ti.current_time();
            std::cout << iterations << ": " << schedule_.top().second
                                     << std::endl;
          }
          if (timeout_ > 0 && ti.current_time() > timeout_) { 
            outoftime = true;
          }
// use the consensus protocol 
#ifdef ASYNCHRONOUS_TERMINATION
          consensus_->begin_critical_section();
          // Receive any inbound messages
          adapter_->receive_bp_messages(this);
          // if I am going to loop, unlock the section
          if (schedule_.size()>0 && schedule_.top().second > bound_ && 
              outoftime == false) {
            consensus_->end_critical_section();
          }
          else {
            if (consensus_->end_critical_section_and_finish()) break;
          }
#else
          // Receive any inbound messages
          size_t numrecv = adapter_->receive_bp_messages(this);
          if (termination_reached() || outoftime) break;
#endif
        } // end of while(finished)


        fout.close();
        std::cout << "Finished Loop to Convergence" << std::endl;
        
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
      
      bool termination_reached() {
        // we send a boolean. true if we are done, false if we are not done
        int32_t sendbuf = (schedule_.top().second <= bound_);
        int32_t recvbuf;
        MPI_Allreduce(&sendbuf, &recvbuf, 1, MPI_INT, MPI_MIN,MPI_COMM_WORLD);
        return recvbuf == 1;
      }

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
        std::cout << mpimessagessent_ << "\n";
        mpibytessent_ = adapter_->numbytessent();
        mpi_inplace_reduce_uint64(mpimessagessent_);
        mpi_inplace_reduce_uint64(numupdates_);
        mpi_inplace_reduce_uint64(numedgeupdates_);
        mpi_inplace_reduce_uint64(boundaryupdates_);
        mpi_inplace_reduce_uint64(boundaryedgeupdates_);
        mpi_inplace_reduce_uint64(interiorupdates_);
        mpi_inplace_reduce_uint64(numops_);
        mpi_inplace_reduce_uint64(mpibytessent_);
      }

      std::map<std::string, double> get_profiling_info(void) const {
        std::map<std::string, double> ret;
        ret["updates"] = numupdates_;
        ret["boundaryupdates"] = boundaryupdates_;
        ret["boundaryedgeupdates"] = boundaryedgeupdates_;
        ret["interiorupdates"] = interiorupdates_;
        ret["edgeupdates"] = numedgeupdates_;
        ret["ops"] = numops_;
        ret["runtime"] = runtime_;
        ret["nummpimessages"] = mpimessagessent_;
        ret["mpibytessent"] = mpibytessent_;
        return ret;
      }

      const factor_type& message(size_t globalsrcvid, size_t globaldestvid) const {
        size_t localsrcvid = adapter_->to_local_vid(globalsrcvid);
        size_t localdestvid = adapter_->to_local_vid(globaldestvid);
        return safe_get(messages_[localsrcvid], localdestvid);
      }
  }; // End of class mpi_wildfire_bp




}; // end of namespace

#include <sill/macros_undef.hpp>

#ifdef ASYNCHRONOUS_TERMINATION
#undef ASYNCHRONOUS_TERMINATION
#endif

#endif // MPI_WILDFIRE_BP

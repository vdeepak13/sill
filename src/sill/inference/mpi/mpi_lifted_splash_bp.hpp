#ifndef MPI_LIFTED_SPLASH_BP_HPP
#define MPI_LIFTED_SPLASH_BP_HPP


// STL includes
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>

// SILL Includes
#include <sill/inference/interfaces.hpp>
#include <sill/mpi/mpi_wrapper.hpp>
#include <sill/mpi/mpi_consensus.hpp>
#include <sill/inference/mpi/mpi_adapter.hpp>
#include <sill/model/lifted_factor_graph_model.hpp>
#include <sill/factor/util/norms.hpp>
#include <sill/math/gdl_enum.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/datastructure/mutable_queue.hpp>
#include <sill/model/factor_graph_partitioning.hpp>

// This include should always be last
#include <sill/macros_def.hpp>

#define HARD_SPLASH_SIZE
namespace sill {
  /**
   Lifted version of mpi_engine
  */
  template<typename F>
  class mpi_lifted_splash_bp:
          public factor_graph_inference<lifted_factor_graph_model<F> > {
    ///////////////////////////////////////////////////////////////////////
    // typedefs
  public:
    typedef F factor_type;
    typedef lifted_factor_graph_model<F> model_type;
    typedef typename lifted_factor_graph_model<F>::variable_type  variable_type;
    typedef typename lifted_factor_graph_model<F>::vertex_type    vertex_type;

    //! IMPORTANT map is in [dest][src] form
    typedef std::map<vertex_type,
                     std::map<vertex_type, F> > message_map_type;

    typedef std::map<vertex_type, F> belief_map_type;
    ///////////////////////////////////////////////////////////////////////
    // Data members
  private:

   
    mpi_post_office &po_;
    mpi_consensus *consensus_;
    mpi_adapter< mpi_lifted_splash_bp<F> > *adapter_;
    
    //! true if I am the root nose
    bool isroot_;
    
    //! pointer to the universe containing all the variables in the factor graph
    universe *universe_;

    //! The root node's pointer to the complete factor graph
    lifted_factor_graph_model<F>* root_factor_graph_;

    //! pointer to the local subset of the factor graph
    lifted_factor_graph_model<F>* local_graph_;
  
    //! the schedule of vertices
    mutable_queue<vertex_type, double> schedule_;

    // messages. Our messages are stored as original_msg
    // this way, the only change we need to make is in the factor->variable equation
    message_map_type messages_;

    //! beliefs
    std::map<vertex_type, F> beliefs_;

    std::map<vertex_type, F> rootbeliefs_;

    //! the size of a splash
    size_t splash_size_;

    //! convergence bound
    double bound_;

    //! level of damping 1.0 is fully damping and 0.0 is no damping
    double damping_;

    //! the commutative semiring for updates (typically sum_product)
    commutative_semiring csr_;

  public:
    size_t boundaryupdates_;
    size_t interiorupdates_;
    size_t numsplashes_;
    size_t totalvertexupdates_;
    
    std::map<vertex_type,int> degreeupdatecount;

    mpi_lifted_splash_bp(mpi_post_office &po): po_(po), csr_(sum_product) {
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
                    lifted_factor_graph_model<F>* factor_graph,
                    size_t splash_size,
                    double bound,
                    double damping,
                    typename factor_graph_partition<F>::algorithm partitionalgorithm =
                        factor_graph_partition<F>::KMETIS,
                    size_t overpartitionfactor = 1) {
      // copy the parameters into the internal fields
      universe_ = universe;
      root_factor_graph_ = factor_graph;
      splash_size_ = splash_size;
      bound_ = bound;
      damping_ = damping;
      isroot_ = true;
      consensus_ = new mpi_consensus(po_);
      adapter_ = new mpi_adapter< mpi_lifted_splash_bp<F> >(po_, *consensus_);
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
        arc << splash_size_ << bound_ << damping_ << root_factor_graph_->weights();
        adapter_->sync_send_all_but_self(strm.str());
      }

      local_graph_ = &(adapter_->get_local_graph());
      update_local_weights(root_factor_graph_->weights());
      create_messages();
      std::cout << po_.id() << " initialized\n";
      std::cout.flush();
      
      po_.barrier();
      
    }

    void initialize_nonroot() {
      isroot_ = false;
      universe_ = new universe;
      consensus_ = new mpi_consensus(po_);
      adapter_ = new mpi_adapter< mpi_lifted_splash_bp<F> >(po_, *consensus_);
      po_.barrier();
      adapter_->initialize_nonroot(*universe_);
      // receive the parameters      
      std::vector<std::map<size_t, size_t> > globalweights;
      po_.barrier();
      {
        std::string str = adapter_->sync_recv();
        std::stringstream strm(str);
        iarchive arc(strm);
        arc >> splash_size_ >> bound_ >> damping_ >> globalweights;
        // we need to reproject the weights to the local vertex ids
      }
      
      local_graph_ = &(adapter_->get_local_graph());
      update_local_weights(globalweights);
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

    void update_local_weights(const std::vector<std::map<size_t,size_t> > &globalweights) {    

      // we need to reproject the weights to the local vertex ids
      foreach(const vertex_type& u, local_graph_->vertices()) {
        if (adapter_->own_vertex(u)) {
          size_t u_id = local_graph_->vertex2id(u);
          size_t g_u_id = adapter_->to_global_vid(u_id);
          foreach(const vertex_type& v, local_graph_->neighbors(u)) {
            size_t v_id = local_graph_->vertex2id(v);
            size_t g_v_id = adapter_->to_global_vid(v_id);
            local_graph_->set_weight(u_id, v_id, safe_get(globalweights[g_u_id], g_v_id));
          }
        }
      } 
    }
    
    void create_messages() {
      // Clear the messages
      messages_.clear();

      // Allocate all messages
      // loop through all the vertices are are truly mine
      foreach(const vertex_type& src, local_graph_->vertices()) {
        if (adapter_->own_vertex(src)) {
          // construct a message to all neighbors
          foreach(const vertex_type& dest, local_graph_->neighbors(src)) {
            // note that messages_[src][dest] will also allocate the messages
            // if it has not been already allocated
            F& msg = messages_[src][dest];
            typename F::domain_type domain = make_domain(dest.is_variable() ?
                                                        &(dest.variable()) :
                                                        &(src.variable()));
            msg = F(domain, 1.0).normalize();

            // construct messages in both directions.
            F& msg2 = messages_[dest][src];
            domain = make_domain(src.is_variable() ?
                                &(src.variable()) :
                                &(dest.variable()));
            msg2 = F(domain, 1.0).normalize();
          }
          // Initialize the belief. also observe that 
          // beliefs_[src] will allocate the beliefs
          F& blf = beliefs_[src];
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
          schedule_.push(v, initial_residual);
        }
      }
    }

    void root_update_belief(vertex_type v, F &b) {
      rootbeliefs_[v] = b;
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
      if(adapter_->own_vertex(target) == false) {
        double delta = norm_1(messages_[source][target], new_msg);
        messages_[source][target] = new_msg;
        adapter_->queue_outgoing_bp_message(source, target, new_msg, delta);
        return;
      }
      assert(messages_.find(source) != messages_.end());
      assert(messages_[source].find(target) != messages_[source].end());
      // Get the original message
      F& original_msg = messages_[source][target];
      
      F prevblf = beliefs_[target];
      F& blf = beliefs_[target];
      double edge_weight = local_graph_->weight(source, target);
      blf.combine_in(pow(original_msg, edge_weight), divides_op);
      blf.combine_in(pow(new_msg,edge_weight), csr_.dot_op);
      blf.normalize();
      //      assert(blf.minimum() > 0.0);
      original_msg = new_msg;


      double new_residual = schedule_.get(target);

      new_residual+=norm_1(prevblf,blf);

      // not sure if this is necessary.
      if (std::isnan(new_residual))
        new_residual = std::numeric_limits<double>::infinity();

      // Update the residual
      //if(schedule_.get(target) < new_residual )
      schedule_.update(target, new_residual);
    } // end of update_message

    inline void send_messages(const vertex_type& source) {

      assert(adapter_->own_vertex(source));
      std::cout.flush();
      //updates_out_ << factor_graph_->vertex2id(source) << std::endl;
      // For each of the neighboring vertices (vertex_target) of
      // vertex compute the message from vertex to vertex_target
      foreach(const vertex_type& dest,
              local_graph_->neighbors(source)) {
        send_message(source, dest);
      }
      // Mark the vertex as having been visited
      schedule_.update(source, 0.0);
      // update the belief
    } // end of update messages



    inline void send_message(const vertex_type& source,
                             const vertex_type& target) {
      // here we can assume that the current belief is correct
      F& blf = beliefs_[source];

      // Construct the cavity
      F cavity = combine(blf, messages_[target][source], divides_op);
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
                                  messages_[source][target],
                                  damping_);
      }
      // update the message also updating the schedule
      update_message(source, target, new_msg);
    } // end of send_message

    void splash(const vertex_type& root) {
      std::vector<vertex_type> splash_order;
      // Grow a splash ordering
      generate_splash(root, splash_order);
      // Push belief from the leaves to the root
      revforeach(const vertex_type& v, splash_order) {
        send_messages(v);
      }
      // Push belief from the root to the leaves (skipping the
      // root which was processed in the previous pass)
      for (size_t i = 1; i < splash_order.size(); ++i) {
        send_messages(splash_order[i]);
      }
    } // End of splash_once


      /**
      * This function computes the splash ordering (a BFS) search for
      * the root vertex
      */

      void generate_splash(const vertex_type& root,
                            std::vector<vertex_type>& splash_order) {
        // Create a set to track the vertices visited in the traversal
        std::set<vertex_type> visited;
        std::list<vertex_type> splash_queue;
        size_t work_in_queue = 0;
        size_t work_in_splash = 0;
        // Set the root to be visited and the first element in the queue
        splash_queue.push_back(root);
        work_in_queue += local_graph_->num_neighbors(root);
        visited.insert(root);
        interiorupdates_++;
        foreach(const vertex_type& v, local_graph_->neighbors(root)) {
          if(adapter_->own_vertex(v)) {
            boundaryupdates_++;
            interiorupdates_--;
            break;
          }
        }
        // Grow a breath first search tree around the root
        for(work_in_splash = 0; work_in_splash < splash_size_
              && !splash_queue.empty();) {
          // Remove the first element
          vertex_type u = splash_queue.front();
          splash_queue.pop_front();
          size_t work = local_graph_->num_neighbors(u);
          #ifdef HARD_SPLASH_SIZE
          if (work + work_in_splash < splash_size_ || work_in_splash == 0) {
          #endif
            splash_order.push_back(u);
            work_in_splash += work;
            work_in_queue -= work;
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
            foreach(const vertex_type& v, local_graph_->neighbors(u)) {
              if((adapter_->own_vertex(v)) && (visited.count(v) == 0)
                && schedule_.get(v) > bound_) {
                splash_queue.push_back(v);
              visited.insert(v);
              work_in_queue += local_graph_->num_neighbors(v);
              }
            } // end of for each neighbors
          }
        } // End of foorloop
        totalvertexupdates_+=work_in_splash;;
        numsplashes_++;
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
        
        boundaryupdates_ = 0;
        interiorupdates_ = 0;
        totalvertexupdates_ = 0;
        numsplashes_ = 0;
        
        MPI::COMM_WORLD.Barrier();
        timer ti;
        ti.start();
        fout << "started " << ti.current_time() << std::endl;
        double lasttime = ti.current_time();
        //assert(schedule_.size() > 0);
        size_t update_count = 0;
        while(1) {
          while(1) {
            consensus_->begin_critical_section();
            // Receive any inbound messages
            adapter_->receive_bp_messages(this);
            // if I am going to loop, unlock the section
            if (schedule_.size()>0 && schedule_.top().second > bound_) {
              consensus_->end_critical_section();
            }
            else {
              break;
            }
            splash(schedule_.top().first);
            update_count++;
            if((ti.current_time() - lasttime) >= 2) {
              lasttime = ti.current_time();
              std::cout << schedule_.top().second << ": "
              << double(boundaryupdates_)/(interiorupdates_+boundaryupdates_)
              << " " << numsplashes_ << " "
              << double(totalvertexupdates_)/numsplashes_ << std::endl;
              boundaryupdates_ = 0;
              interiorupdates_ = 0;
              totalvertexupdates_ = 0;
              numsplashes_ = 0;
            }
           
            // Send newly created outbound messages
            adapter_->flush(this, false);
          }  // While great than bound
          
          // There could be a race here as new messages may come in at
          // this point, and call the handler, before I can call finish()
          // Send any remaining messages
          adapter_->flush(this);
          
//          std::cout << "stopped " << ti.current_time() << std::endl;
          // Check termination
          if (consensus_->end_critical_section_and_finish()) break;
//          std::cout << "started " << ti.current_time() << std::endl;
          adapter_->receive_bp_messages(this);
        } // end of while(finished)
//        std::cout<< "stopped " << ti.current_time() << std::endl;
        fout.close();
        std::cout << "Finished Splash to Convergence" << std::endl;
        
        double t = ti.current_time();
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
        return t;
      } // End of run


      const F& belief(variable_type variable) const {
        return belief(vertex_type(variable));
      } // end of send_message

      const F& belief(const vertex_type& vert) const {
        assert(isroot_);
        assert(rootbeliefs_.count(vert));
        return rootbeliefs_.find(vert)->second;
      } // end of send_message


      std::map<vertex_type, F> belief() const {
        assert(isroot_);
        return rootbeliefs_;
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
      
      const factor_type& message(size_t globalsrcvid, size_t globaldestvid) const {
        size_t localsrcvid = adapter_->to_local_vid(globalsrcvid);
        size_t localdestvid = adapter_->to_local_vid(globaldestvid);
        return safe_get(safe_get(messages_, id2vertex(localsrcvid)), id2vertex(localdestvid));
      }
  }; // End of class mpi_lifted_splash_bp




}; // end of namespace

#ifdef HARD_SPLASH_SIZE
#undef HARD_SPLASH_SIZE
#endif
#include <sill/macros_undef.hpp>



#endif // MPI_LIFTED_SPLASH_BP_HPP

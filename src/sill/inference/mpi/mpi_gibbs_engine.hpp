#ifndef MPI_GIBBS_ENGINE_HPP
#define MPI_GIBBS_ENGINE_HPP


// STL includes
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>

// SILL Includes
#include <sill/model/factor_graph_model.hpp>
#include <sill/factor/util/norms.hpp>
#include <sill/math/gdl_enum.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/datastructure/mutable_queue.hpp>

#include <sill/inference/mpi/mpi_gibbs_adapter.hpp>


// This include should always be last
#include <sill/macros_def.hpp>
namespace sill {

  template<typename F>
  class mpi_gibbs_engine {
    ///////////////////////////////////////////////////////////////////////
    // typedefs
  public:
    typedef F factor_type;
    typedef factor_type message_type;
    typedef factor_type belief_type;
    typedef typename factor_type::domain_type domain_type;

    typedef factor_graph_model<factor_type>     factor_graph_type;
    typedef typename factor_graph_type::variable_type    variable_type;
    typedef typename factor_graph_type::vertex_type      vertex_type;

    typedef factor_graph_partition<factor_graph_type> partition_type;
    typedef typename partition_type::algorithm partition_algorithm_type;


    typedef std::set<vertex_type> vertex_set_type;


    typedef std::set<const factor_type*> factor_set_type;

    typedef std::map<variable_type, factor_set_type> var2factors_type;

    typedef factor_norm_inf_log<message_type> norm_type;

    typedef std::map<vertex_type, belief_type> belief_map_type;

    typedef mpi_gibbs_adapter< mpi_gibbs_engine<F> > adapter_type;
   
    ///////////////////////////////////////////////////////////////////////
    // Data members
  private:    

    //! A communication adapater
    adapter_type adapter_;

    //! pointer to the factor graph 
    factor_graph_type* factor_graph_;
    
    //! random number generator
    boost::mt11213b rng_;

    //! uniform distribution over [0, 1]
    boost::uniform_real<double> uniform_real_;

    //! estimate of the beliefs
    belief_map_type beliefs_;
    
    //! map from variable to factors (by pointer)
    var2factors_type var2factors;
   
    //! current sample
    finite_assignment cur_sample_;

    //! number of iterations
    size_t iterations_;

  public:

    mpi_gibbs_engine(factor_graph_type* factor_graph,
                     size_t iterations,
                     double a,
                     double b,
                     partition_algorithm_type partitionalgorithm = 
                      partition_type::PMETIS,
                     int overpartitionfactor = 1) :
      factor_graph_(factor_graph), 
      iterations_(iterations) {
      // Transmit / receive the factor graph
      if(adapter_.id() == adapter_type::ROOT_NODE) {
        adapter_.send_factor_graph(factor_graph_);
      } else {
        factor_graph_ = adapter_.recv_factor_graph();
      }
    } // end mpi_gibbs_engine

    void run(double* runningtime = NULL) {
      srand(time(NULL) + adapter_.id());
      initialize_state();
      run_to_convergence();
      // Broadcast beliefs too root
      adapter_.sync_beliefs(beliefs_);
    } // End of run



    void initialize_state() {
      // Initialize the variable to factor map
      foreach(const factor_type& factor, factor_graph_->factors()) {
        foreach(finite_variable* v, factor.arguments()) {
          var2factors[v].insert(&factor);
        }
      }
      // Initialize the beliefs 
      foreach(const vertex_type& u, factor_graph_->vertices()) {
        belief_type& blf = beliefs_[u];
        if(u.is_factor()) { 
          blf = belief_type(u.factor().arguments(),0.0);
        } else {
          blf = belief_type(make_domain(&(u.variable())),0.0);
        }
        if (u.is_factor()) {
          finite_assignment mapassg = arg_max(u.factor());
          foreach(finite_variable* v, u.factor().arguments()) {
            cur_sample_[v] = mapassg[v];
          }
        }
      }
      // Initialize the first assignment
      foreach(finite_variable* v, factor_graph_->arguments()) {
        boost::uniform_int<size_t> unifint(0, v.size());
        size_t r = unifint(rng_);
        cur_sample_[v] = r;
      }
    } // end of initialize

    void run_to_convergence() {
      const int BURNIN= 100;
      if (iterations_ < BURNIN) {
        std::cout << "#iterations < burnin" << std::endl;
        return;
      }
      for(size_t i = 0; i < iterations_; ++i) {
        std::cout << "Sample: " << i << std::endl;
        sample_once();
        if (i>BURNIN) {
          //        print_cur_sample();
          std::cout << "Update Blf: " << i << std::endl;
          update_beliefs();
        }
      }
    } // end of splash_to_convergence


    void print_cur_sample() {
      typedef finite_assignment::value_type pair;
      foreach(const pair& p, cur_sample_) {
        std::cout << p.second << " ";
      }
      std::cout << std::endl;
    }

    //! update the current sample
    void sample_once() {
      size_t var_count = 0;
      foreach(variable_type var, factor_graph_->arguments()) {
        factor_type conditional = 1.0;
        foreach(const factor_type* factor, var2factors[var]) {
          // Construct the conditional assignment to the other variables
          finite_assignment cond_asg;
          foreach(variable_type other_var, factor->arguments()) {
            if(other_var != var){
              cond_asg[other_var] = cur_sample_[other_var];
            }
          }
          // Restrict this factor and multiply into conditional
          factor_type restrictedfactor = factor->restrict(cond_asg);
          restrictedfactor.normalize();
          conditional.combine_in(restrictedfactor, product_op);
          conditional.normalize();
        }
        // Draw a random sample
        cur_sample_[var] = sample(conditional);
      }
        std::cout << "Progress. " << std::endl;
    }

    //! draw a random sample from a factor
    size_t sample(factor_type& factor) {
      assert(factor.arguments().size() == 1);
      factor.normalize();
      double sum = 0;
      double r = uniform_real_(rng_);
      size_t index = 0;
      foreach(double d, factor.values()) {
        sum += d;
        if(r <= sum) return index;
        else index++;
      }
      assert(false);
      return -1;
    } // end of sample from a factor
    
    //! update the belief counts
    void update_beliefs() {
      typedef typename belief_map_type::value_type pair_type;
      size_t var_count = 0;
      
      foreach(pair_type& pair, beliefs_) {
      //  double progress = 100.0*(var_count++) / beliefs_.size();
       // if( size_t(progress) != last_progress ) {
       //   std::cout << "Progress: " << progress << std::endl;
       // }

        // Construct the conditional assignment to the other variables
        belief_type& blf = pair.second;
        finite_assignment full_asg;
        foreach(variable_type var, blf.arguments()) {
          full_asg[var] = cur_sample_[var];
        }
        typename factor_type::result_type &f = blf(full_asg);
        f = f + 1.0;
      }
    } // end of update beliefs

    /**
     * Compute the belief for a vertex
     */
    const belief_type& belief(variable_type variable) {
      return beliefs_[vertex_type(variable)];
    } // end of send_message

    void belief(std::map<vertex_type, belief_type> &beliefs) {
      beliefs = beliefs_;
    } // end of send_message

   void get_map_assignment(finite_assignment &mapassg) {
      foreach(const vertex_type &v, factor_graph_->vertices()) {
        if (v.is_variable()) {
          finite_assignment localmapassg = arg_max(beliefs_[v]);
          mapassg[&(v.variable())] = localmapassg[&(v.variable())];
        }
      }
    }
 

  }; // end of gibbs engine
    
}; // end of namespace
#include <sill/macros_undef.hpp>



#endif

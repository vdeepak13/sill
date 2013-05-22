#ifndef GIBBS_ENGINE_HPP
#define GIBBS_ENGINE_HPP


// STL includes
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>

// Boost random number libraries
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/lagged_fibonacci.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/foreach.hpp>


// PRL Includes
#include <sill/model/factor_graph_model.hpp>
#include <sill/factor/norms.hpp>
#include <sill/math/gdl_enum.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/datastructure/mutable_queue.hpp>
#include <sill/inference/bp_convergence_measures.hpp>



// This include should always be last
#include <sill/macros_def.hpp>
namespace sill {

  template<typename F>
  class gibbs_engine {
    ///////////////////////////////////////////////////////////////////////
    // typedefs
  public:
    typedef F factor_type;
    typedef factor_type belief_type;
    typedef typename factor_type::domain_type domain_type;

    typedef factor_graph_model<factor_type>     factor_graph_type;
    typedef typename factor_graph_type::variable_type    variable_type;
    typedef typename factor_graph_type::vertex_type      vertex_type;

    typedef std::set<vertex_type> vertex_set_type;

    typedef std::map<vertex_type, belief_type> belief_map_type;

    typedef std::set<factor_type*> factor_set_type;

    typedef std::map<variable_type*, factor_set_type> var2factors_type;

    ///////////////////////////////////////////////////////////////////////
    // Data members
  private:
    //! pointer to the factor graph
    factor_graph_type* factor_graph_;

    //! random number generator
    boost::lagged_fibonacci607 rng_;

    //! uniform distribution over [0, 1]
    boost::uniform_real<double> uniform_real_;

    //! estimate of the beliefs
    belief_map_type beliefs_;

    //! map from variable to factors (by pointer)
    var2factors_type var2factors;

    //! current sample
    finite_assignment cur_sample_;

    size_t iterations_;

    //! this object tells us when we're done
    residual_splash_convergence_measure* convergence_indicator_;

    //! largest cardinality we expect from the variables
    size_t largest_var_cardinality_;

    //! max domain size of pre-merged factos used to compute the conditional
    const size_t MAX_JOIN_FACTOR_DOMAIN_SIZE_;

  public:
    gibbs_engine(factor_graph_type* factor_graph,
                 size_t iterations,
                 residual_splash_convergence_measure* convergence_indicator,
                 double b) :
      factor_graph_(factor_graph),
      rng_(time(NULL)),
      iterations_(iterations),
      convergence_indicator_(convergence_indicator),
      MAX_JOIN_FACTOR_DOMAIN_SIZE_(10)
      {
      // std::cout << uniform_real_(rng_)<<" , "  << uniform_real_(rng_) << "\n";
      // Ensure that the factor graph is not null
      assert(factor_graph_ != NULL);

      largest_var_cardinality_ = 0;
      foreach(const finite_variable* v, factor_graph_->arguments())
        largest_var_cardinality_ = std::max((size_t) largest_var_cardinality_, v->size());

    } // end gibbs_engine

    ~gibbs_engine(){
      foreach(finite_variable* v, factor_graph_->arguments())
        foreach(factor_type* f, var2factors[v])
          delete f;
    }

    void run() {
      initialize_state();
      run_to_convergence();
    } // End of run

    void initialize_state() {
      // Initialize the variable to factor map
      foreach(const factor_type& factor, factor_graph_->factors()) {
        foreach(finite_variable* v, factor.arguments()) {
          bool combined_in = false;
          foreach(factor_type* inner_factor, var2factors[v]){
            if(set_union(inner_factor->arguments(), factor.arguments()).size() < MAX_JOIN_FACTOR_DOMAIN_SIZE_){
              inner_factor->combine_in(factor, sill::product_op);
              combined_in = true;
              break;
            }
          }
          if(!combined_in){
            factor_type* f = new factor_type(factor);
            var2factors[v].insert(f);
          }
        }
      }
      // Initialize the state randomly
      foreach(finite_variable* v, factor_graph_->arguments()){
        belief_type blf(make_domain(v), 1.0);
        cur_sample_[v] = sample(blf);
        beliefs_[vertex_type(v)] = belief_type(make_domain(v),0.0);
      }
    } // end of initialize

    void run_to_convergence() {
      convergence_indicator_->start();

      const size_t BURNIN = iterations_/ 4;
      for(size_t i = 0; i < BURNIN; ++i) {
        sample_once();
        if (i % 100 == 0) {
          std::cout << "burnin: " << i << std::endl;
        }
      }
      size_t samples_count = 0;
      while(!convergence_indicator_->
            is_converged(iterations_ - samples_count, 
                         samples_count)){
        sample_once();
        update_beliefs();
        samples_count++;
        if (samples_count % 100 == 0) {
          std::cout << "samples: " << samples_count << std::endl;
        }
      }

      std::cout << "total samples: " << samples_count << std::endl;

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

      double probas[largest_var_cardinality_];
      typename factor_type::result_type 
        restricted_factor_values[largest_var_cardinality_];

      BOOST_FOREACH(variable_type* var, factor_graph_->arguments()) {
        assert(var->size() <= largest_var_cardinality_);

        for(size_t i=0; i<var->size(); i++)
          restricted_factor_values[i] = typename factor_type::result_type(1.0);

        size_t* var_value = &cur_sample_[var];
        BOOST_FOREACH(const factor_type* factor, var2factors[var]) {
          //manual restrict + combine
          for(*var_value=0; *var_value < var->size(); (*var_value)++)
            restricted_factor_values[*var_value] *= factor->v(cur_sample_);
        }

        typename factor_type::result_type max_val = restricted_factor_values[0];
        for(size_t i=0; i<var->size(); i++)
          max_val = std::max(max_val, restricted_factor_values[i]);

        for(size_t i=0; i<var->size(); i++)
          probas[i] = (restricted_factor_values[i] / max_val);

        // Draw a random sample
        *var_value = sample(probas, var->size());
      }
    }

    //! draw a random sample from a factor
     size_t sample(factor_type& factor) {
       assert(factor.arguments().size() == 1);
       double sum = 0;
       double r = uniform_real_(rng_) * factor.norm_constant();
       size_t index = 0;
       foreach(double d, factor.values()) {
         sum += d;
         if(r <= sum) return index;
         else index++;
       }
       assert(false);
       return -1;
     } // end of sample from a factor

    //! draw a random sample from a factor
    size_t sample(double* probas_unnormalized, size_t cardinality) {

      double norm_constant = 0;
      for(size_t i=0; i<cardinality; i++)
        norm_constant += probas_unnormalized[i];

      double r = uniform_real_(rng_) * norm_constant;
      double sum = 0;
      for(size_t i=0; i<cardinality; i++){
        sum += probas_unnormalized[i];
        if(r <= sum){
          return i;
        }
      }

      assert(false);
      return -1;
    } // end of sample from a factor

    //! update the belief counts
    void update_beliefs() {
      typedef typename belief_map_type::value_type pair_type;
      // size_t var_count = 0;

      foreach(pair_type& pair, beliefs_) {
      //  double progress = 100.0*(var_count++) / beliefs_.size();
       // if( size_t(progress) != last_progress ) {
       //   std::cout << "Progress: " << progress << std::endl;
       // }

        // Construct the conditional assignment to the other variables
        belief_type& blf = pair.second;
        finite_assignment full_asg;
        foreach(variable_type* var, blf.arguments()) {
          full_asg[var] = cur_sample_[var];
        }
        typename factor_type::result_type f = blf.v(full_asg);
        f = f + 1.0;
        blf.set_v(full_asg, f);
      }
    } // end of update beliefs

    /**
     * Compute the belief for a vertex
     */
    belief_type belief(variable_type* variable) {
      belief_type b = beliefs_[vertex_type(variable)];
      b.normalize();
      return b;
    }

    void belief(std::map<vertex_type, belief_type> &beliefs) {
      beliefs = beliefs_;
    }

   void map_assignment(finite_assignment &mapassg) {
      foreach(const vertex_type &v, factor_graph_->vertices()) {
        if (v.is_variable()) {
          finite_assignment localmapassg = arg_max(beliefs_[v]);
          mapassg[&(v.variable())] = localmapassg[&(v.variable())];
        }
      }
    }

  }; // End of class gibbs_engine




}; // end of namespace
#include <sill/macros_undef.hpp>


#endif /* GIBBS_ENGINE_HPP */

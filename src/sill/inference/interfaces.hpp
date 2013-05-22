#ifndef SILL_INFERENCE_INTERFACES_HPP
#define SILL_INFERENCE_INTERFACES_HPP

#include <sill/base/timed_process.hpp>
#include <sill/factor/concepts.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Forward declaration
  template <typename F> class dynamic_bayesian_network;
  
  //! An interface that represents a filter
  //! \ingroup inference
  template <typename F>
  class filter {
    concept_assert((DistributionFactor<F>));

  public:
    typedef typename F::variable_type variable_type;

    typedef typename F::domain_type domain_type;

    typedef timed_process<variable_type> process_type;

  public:
    //! Destructor
    virtual ~filter() {}

    //! Returns the dynamic Bayesian network associated with this filter
    virtual const dynamic_bayesian_network<F>& model() const = 0;

    //! Advances the state to the next time step
    virtual void advance() = 0;

    //! Multiplies in the likelihood to the belief state
    virtual void estimation(const F& likelihood) = 0;
    
    //! Extracts the beliefs over a subset of the processes
    virtual F belief(const std::set<process_type*>& processes) const = 0;

    //! Extracts the beliefs over a subset of step-t variables
    virtual F belief(const domain_type& variables) const = 0;

    //! Extracts the belief over a single step-t variable
    virtual F belief(variable_type* v) const = 0;
  };


  //! An interface that represents a factor graph inference engine
  //! \ingroup inference
  template <typename Model>
  class factor_graph_inference {

    typedef typename Model::factor_type factor_type;
    typedef typename Model::vertex_type vertex_type;
    typedef typename factor_type::variable_type variable_type;

  public:
    //! Destructor
    virtual ~factor_graph_inference() {}

    //! Clears the inference engine
    virtual void clear() = 0;

    //! Loops until converegence
    virtual double loop_to_convergence() = 0;

    //! Returns the belief of a particular variable
    virtual const factor_type& belief(variable_type* variable) const = 0;
    
    //! Returns the belief of a particular vertex in the factor graph
    virtual const factor_type& belief(const vertex_type& vert) const = 0;

    //! Returns the message from src vertex id to dest vertex id. Only 
    //! applicable to the BP class of algorithms
    virtual const factor_type& message(size_t globalsrcvid, size_t globaldestvid) const = 0;

    //! Returns the belief of all vertices in the factor graph
    virtual std::map<vertex_type, factor_type> belief() const = 0;

    //! Returns the map assignment over all variables
    virtual void map_assignment(finite_assignment &mapassg) const = 0;

    //! Optional. Returns some profiling information about the last inference call
    virtual std::map<std::string, double> get_profiling_info(void) const {
      std::map<std::string, double> ret;
      return ret;
    }
  };

};

#include <sill/macros_undef.hpp>

#endif

#ifndef SILL_INFERENCE_INTERFACES_HPP
#define SILL_INFERENCE_INTERFACES_HPP

#include <sill/base/discrete_process.hpp>
#include <sill/factor/concepts.hpp>

#include <sill/macros_def.hpp>

namespace sill {


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

#ifndef SILL_MEAN_FIELD_PAIRWISE_HPP
#define SILL_MEAN_FIELD_PAIRWISE_HPP

#include <sill/global.hpp>
#include <sill/boost_unordered_utils.hpp>
#include <sill/model/markov_network.hpp>

#include <boost/unordered_map.hpp>

#include <functional>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A class that runs the mean field algorithm for a pairwise Markov
   * network. The computation is performed sequentially in the order
   * of the vertices in Markov network.
   * 
   * \tparam F the type that represents the factor associated
   *           with vertices and edges. Typically, this type
   *           represents the canonical parameterization of
   *           the distribution.
   */
  template <typename F>
  class mean_field_pairwise {
  public:
    // factor-related typedefs
    typedef typename F::real_type               real_type;
    typedef typename F::probability_factor_type belief_type;

    // model-related typedefs
    typedef pairwise_markov_network<F> model_type;
    typedef typename F::variable_type  variable_type;
    typedef typename model_type::edge  edge_type;
    
    /**
     * Creates a mean field engine for the given graph.
     * The graph vertices must not change after initialization
     * (the potentials may).
     */
    explicit mean_field_pairwise(const model_type* model)
      : model_(*model) {
      foreach(variable_type* v, model_.vertices()) {
        beliefs_[v] = belief_type(make_domain(v)).normalize();
      }
    }

    /**
     * Performs a single iteration of mean field.
     */
    real_type iterate() {
      real_type sum = 0.0;
      foreach (variable_type* v, model_.vertices()) {
        sum += update(v);
      }
      return sum / model_.num_vertices();
    }

    /**
     * Returns the belief for a vertex.
     */
    const belief_type& belief(variable_type* v) const {
      return get(beliefs_, v);
    }

  private:
    /**
     * Updates a single vertex.
     */
    real_type update(variable_type* v) {
      F result = model_[v];
      foreach (edge_type e, model_.in_edges(v)) {
        model_[e].log_exp_mult(belief(e.source()), result);
      }
      belief_type new_belief(result);
      new_belief.normalize();
      get(beliefs_, v).swap(new_belief);
      return diff_1(new_belief, belief(v));
    }
    
    //! The underlying graphical model
    const model_type& model_;

    //! A map of current beliefs, one for each variable
    boost::unordered_map<variable_type*, belief_type> beliefs_;

  };

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

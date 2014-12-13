#ifndef SILL_ASYNCHRONOUS_GBP_PC_HPP
#define SILL_ASYNCHRONOUS_GBP_PC_HPP

#include <sill/inference/loopy/gbp_pc.hpp>
#include <sill/graph/algorithm/graph_traversal.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * GBP engine that updates the messages in a topological order.
   * according to the natural order of messages.
   */
  template <typename F>
  class asynchronous_gbp_pc : public gbp_pc<F> {

  public:
    typedef gbp_pc<F> base;
    typedef typename F::variable_type variable_type;
    typedef typename F::domain_type   domain_type;
    typedef typename base::edge       edge;

  protected:
    using base::graph;
    using base::pass_message;

  public:
    asynchronous_gbp_pc(const region_graph<variable_type*,F>& graph)
      : base(graph) { }
    
    double iterate(double eta) {
      double residual = 0;
      // pass the messages downwards
      foreach(edge e, graph.edges())
        residual = std::max(residual, pass_message(e, eta));
      return residual;
    }
    
  }; // class asynchronous_gbp_pc

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

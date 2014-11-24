#ifndef SILL_ASYNCHRONOUS_GBP_HPP
#define SILL_ASYNCHRONOUS_GBP_HPP

#include <sill/inference/loopy/gbp.hpp>
#include <sill/graph/graph_traversal.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * GBP engine that updates the messages in a topological order.
   * according to the natural order of messages.
   */
  template <typename F>
  class asynchronous_gbp : public gbp<F> {

  public:
    typedef gbp<F> base;
    typedef typename F::variable_type variable_type;
    typedef typename F::domain_type   domain_type;

    //! The update order
    std::vector<size_t> order;

  protected:
    using base::graph;
    using base::pass_message;

  public:
    asynchronous_gbp(const region_graph<variable_type*,F>& graph)
      : base(graph), order(directed_partial_vertex_order(graph)) { }
    
    double iterate(double eta) {
      // pass the messages downwards
      double residual = 0.0;
      foreach(size_t v, order) {
        foreach(size_t u, graph.parents(v)) {
          residual = std::max(residual, pass_message(u, v, eta));
        }
      }
      // pass the messages upwards
      foreach(size_t v, make_reversed(order)) {
        foreach(size_t u, graph.parents(v)) {
          residual = std::max(residual, pass_message(v, u, eta));
        }
      }
      return residual;
    }
    
  }; // class asynchronous_gbp

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

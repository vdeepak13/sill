#ifndef SILL_RESIDUAL_BETHE_BP
#define SILL_RESIDUAL_BETHE_BP

#include <sill/inference/bethe_bp.hpp>
#include <sill/datastructure/mutable_queue.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Loopy BP engine that implements residual B.P.
   *
   * \ingroup inference
   */
  template <typename F>
  class residual_bethe_bp : public bethe_bp<F> {

  public:
    typedef bethe_bp<F> base;
    typedef typename base::vertex     vertex;
    typedef typename base::edge       edge; 
    typedef typename F::variable_type variable_type;
    typedef typename F::domain_type   domain_type;

  protected:
    // shortcuts from the base
    using base::pass_flow;
    using base::g;
    using base::message;
    using base::compute_message;

    mutable_queue<edge, double> q; //< The queue of weights

    //! Updates the residuals for an edge
    void update_residual(edge e) {
      double r = norm_inf(message(e), compute_message(e));
      if (!q.contains(e)) q.push(e, r); else q.update(e, r);
    }

  public:
    residual_bethe_bp(const factorized_model<F>& model)
      : base(model) { 
      foreach(edge e, g.edges()) update_residual(e);
    }

    residual_bethe_bp(const factorized_model<F>& model,
                      const std::vector<domain_type>& clusters)
      : base(model, clusters) { 
      foreach(edge e, g.edges()) update_residual(e);
    }

    size_t add_factor(const F& factor) {
      size_t v = base::add_factor(factor);
      foreach(edge e, g.out_edges(v)) update_residual(e);
      foreach(edge e, g.in_edges(v)) update_residual(e);
      return v;
    }
    
    //! Performs one round of updates with the given learning rate
    double iterate(double eta = 1) {
      if (!q.empty()) {
        // extract the leading candidate edge
        edge e; double r;
        boost::tie(e, r) = q.pop();

        // pass the flow and update dependent messages
        // cout << "Passing flow between " << vp << " r=" << r << endl;
        double residual = pass_flow(e, eta);
        foreach(edge e_update, g.out_edges(e.target())) {
          if (e_update.target() != e.source()) 
            update_residual(e_update);
        }
        if (eta<1) update_residual(e);
        return residual;
      } else return 0;
    }

  };

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

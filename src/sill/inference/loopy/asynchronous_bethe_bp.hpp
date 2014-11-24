#ifndef SILL_ASYNCHRONOUS_BETHE_BP
#define SILL_ASYNCHRONOUS_BETHE_BP

#include <sill/inference/loopy/bethe_bp.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Loopy BP engine with an asynchronous ordering of updates.
   * \ingroup inference
   */
  template <typename F>
  class asynchronous_bethe_bp : public bethe_bp<F> {

  public:
    typedef bethe_bp<F> base;
    typedef typename base::edge edge;
    typedef typename F::variable_type variable_type;
    typedef typename F::domain_type domain_type;

  protected:
    using base::g;
    using base::pass_flow;

  public:
    asynchronous_bethe_bp(const factorized_model<F>& model)
      : base(model) { }

    asynchronous_bethe_bp(const factorized_model<F>& model,
                          const std::vector<domain_type>& clusters)
      : base(model, clusters) { }
    
    //! Performs one round of updates with the given learning rate
    double iterate(double eta = 1) {
      double error = 0;
      // iterate in some arbitrary order (as given by the graph)
      foreach(edge e, g.edges()) {
        double residual = pass_flow(e, eta);
        //        if (residual > error) 
//         if (residual >= 1e8) {
//           std::cout << residual << " : "
//                     << g[e.source()].arguments() << " -> " 
//                     << g[e.target()].arguments() << std::endl;
//           std::cout << g[e] << std::endl;
//           std::cout << g[e.source()] << std::endl;
//         }
        
        error = std::max(residual, error);
      }
      return error;
    }

  }; // class asynchronous_bethe_bp

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

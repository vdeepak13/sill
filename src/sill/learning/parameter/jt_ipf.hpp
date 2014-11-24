#ifndef SILL_IPF_HPP
#define SILL_IPF_HPP

#include <boost/unordered_map.hpp>

#include <sill/global.hpp>
#include <sill/factor/concepts.hpp>
#include <sill/factor/norms.hpp>
#include <sill/model/decomposable.hpp>
#include <sill/inference/exact/junction_tree_inference.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * An engine that performs parameter learning with iterated proportional
   * fitting.
   * \ingroup learning_param
   */
  template <typename F>
  class jt_ipf {
    concept_assert((Factor<F>));

    // Public type declarations
    // =========================================================================
  public:
    typedef size_t vertex;

    typedef typename F::variable_type variable_type;

    // Private type declarations and data members
    // =========================================================================
  private:
    //! Norm used to evaluate the convergence (owned by this engine)
    factor_norm<F>& norm;

    //! The junction tree engine that stores the potentials and the messages
    hugin<F> jt_engine;

    //! Node to marginals
    boost::unordered_map< vertex, std::vector<F> > marginals;

    // Constructors and destructors
    // =========================================================================
  public:

    //! Constructor
    /*
    jt_ipf(const junction_tree<variable_type*>& jt)
      :jt_engine(jt) { }
    */

    //! Constructor
    jt_ipf(const std::vector<F>& marginals,
           const factor_norm<F>& norm = factor_norm_inf<F>())
      : norm(*norm.clone()), jt_engine(marginals) {
      foreach(const F& marginal, marginals) {
        vertex v = 
          jt_engine.tree().find_clique_cover(marginal.arguments());
        this->marginals[v].push_back(marginal);
      }
      jt_engine.calibrate();
    }

    //! Destructor
    ~jt_ipf() {
      delete &norm;
    }

    //! Performs a number of iterations over all marginals
    //! \return the error for the latest iteration
    double iterate(size_t n) {
      double error = 0;
      for(size_t i = 0; i < n; i++) {
        vertex root = jt_engine.tree().root();
        error = visit(root);
        //jt_engine.distribute_evidence(root);
        jt_engine.calibrate();
      }
      return error;
    }
    
    //! Returns the current estimate of the distribution as a decomposable model
    decomposable<F> result() {
      jt_engine.normalize();
      return decomposable<F>(jt_engine.clique_beliefs());
    }

    // Private member functions
    // =========================================================================
  private:
    //! Performs update at vertex v
    double update(vertex v) {
      double error = 0;
      foreach(const F& factor, marginals[v]) {
        F current = jt_engine.potential(v).marginal(factor.arguments());
        jt_engine.potential(v) /= current;
        jt_engine.potential(v) *= factor;
        error = std::max(norm(current, factor), error);
      }
      return error;
    }

    //! A recursive procedure that updates u and all vertices away from parent
    double visit(vertex u, vertex parent = vertex()) {
      double error = update(u);
      foreach(vertex v, jt_engine.tree().neighbors(u)) {
        if (v != parent) {
          jt_engine.pass_flow(u, v);
          error = std::max(visit(v, u), error);
          jt_engine.pass_flow(v, u);
        }
      }
      return error;
    }

  }; // class jt_ipf

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

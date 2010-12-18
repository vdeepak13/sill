#ifndef SILL_LEARN_DECOMPOSABLE
#define SILL_LEARN_DECOMPOSABLE

#include <sill/learning/dataset/dataset.hpp>
#include <sill/learning/learn_factor.hpp>
#include <sill/model/decomposable.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Given a junction tree structure, learn a decomposable model by estimating
   * probabilities over cliques from the given dataset.
   * @todo Permit cross-validation to choose smoothing.
   */
  template <typename F>
  void learn_decomposable(decomposable<F>& model,
                          const typename decomposable<F>::jt_type& structure,
                          const dataset& ds, double smoothing) {
    model.clear();
    std::vector<F> marginals;
    foreach(const typename decomposable<F>::vertex& v, structure.vertices()) {
      marginals.push_back
        (learn_marginal<F>(structure.clique(v), ds, smoothing));
    }
    model.initialize(structure, marginals);

  } // learn_decomposable

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_LEARN_DECOMPOSABLE

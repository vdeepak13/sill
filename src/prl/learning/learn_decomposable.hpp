#ifndef PRL_LEARN_DECOMPOSABLE
#define PRL_LEARN_DECOMPOSABLE

#include <prl/learning/dataset/dataset.hpp>
#include <prl/learning/learn_factor.hpp>
#include <prl/model/decomposable.hpp>

#include <prl/macros_def.hpp>

namespace prl {

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

} // namespace prl

#include <prl/macros_undef.hpp>

#endif // PRL_LEARN_DECOMPOSABLE


#ifndef PRL_GENERATE_DATASETS_HPP
#define PRL_GENERATE_DATASETS_HPP

/**
 * \file generate_datasets.hpp  Free functions for generating synthetic data.
 */

#include <prl/learning/dataset/vector_assignment_dataset.hpp>
#include <prl/model/crf_model.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  /**
   * Generate n samples over X from the given model for P(X),
   * and insert them into the given dataset.
   * Any samples already in the dataset are kept.
   */
  template <typename F, typename RandomNumGen>
  void generate_dataset(dataset& ds,
                        const decomposable<F>& model,
                        size_t n, RandomNumGen& rng) {
    for (size_t i(0); i < n; ++i) {
      ds.insert(model.sample(rng));
    }
  }

  /**
   * Generate n samples over Y,X from the given models for P(X) and P(Y|X),
   * and insert them into the given dataset.
   * Any samples already in the dataset are kept.
   */
  template <typename F, typename CRFfactor, typename RandomNumGen>
  void generate_dataset(dataset& ds,
                        const decomposable<F>& Xmodel,
                        const crf_model<CRFfactor>& YgivenXmodel,
                        size_t n, RandomNumGen& rng) {
    for (size_t i(0); i < n; ++i) {
      typename F::assignment_type fa(Xmodel.sample(rng));
      typename CRFfactor::output_assignment_type
        fa2(YgivenXmodel.sample(fa, rng));
      ds.insert(assignment(map_union(fa, fa2)));
    }
  }

} // namespace prl

#include <prl/macros_undef.hpp>

#endif // #ifndef PRL_GENERATE_DATASETS_HPP

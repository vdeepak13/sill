#ifndef SILL_GENERATE_DATASETS_HPP
#define SILL_GENERATE_DATASETS_HPP

/**
 * \file generate_datasets.hpp  Free functions for generating synthetic data.
 */

#include <iostream>

#include <sill/learning/dataset/oracle.hpp>
#include <sill/learning/dataset/vector_assignment_dataset.hpp>
#include <sill/model/bayesian_network.hpp>
#include <sill/model/crf_model.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Generate n samples over X from the given model for P(X),
   * and insert them into the given dataset.
   * Any samples already in the dataset are kept.
   */
  template <typename LA, typename F, typename RandomNumGen>
  void generate_dataset(dataset<LA>& ds,
                        const decomposable<F>& model,
                        size_t n, RandomNumGen& rng) {
    for (size_t i(0); i < n; ++i) {
      ds.insert(model.sample(rng));
    }
  }

  /**
   * Generate n samples over X from the given model for P(X),
   * and insert them into the given dataset.
   * Any samples already in the dataset are kept.
   */
  template <typename LA, typename F, typename RandomNumGen>
  void generate_dataset(dataset<LA>& ds,
                        const bayesian_network<F>& model,
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
  template <typename LA, typename F, typename CRFfactor, typename RandomNumGen>
  void generate_dataset(dataset<LA>& ds,
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

  /**
   * Draw examples from the given oracle to build a dataset.
   */
  template <typename LA1, typename LA2>
  void
  oracle2dataset(oracle<LA1>& o, size_t max_records, dataset<LA2>& ds) {
    ds.reset(o.datasource_info());
    for (size_t i = 0; i < max_records; i++) {
      if (o.next())
        ds.insert(o.current().finite(), o.current().vector());
      else
        break;
    }
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_GENERATE_DATASETS_HPP

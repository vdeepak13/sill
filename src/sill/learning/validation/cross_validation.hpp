#ifndef SILL_CROSS_VALIDATION_HPP
#define SILL_CROSS_VALIDATION_HPP

#include <sill/learning/dataset/slice.hpp>

#include <vector>

namespace sill {

  template <typename Dataset>
  void kfold_split(Dataset& ds,
                   size_t num_folds,
                   std::vector<typename Dataset::slice_view_type>& train,
                   std::vector<typename Dataset::slice_view_type>& test) {
    size_t size = ds.size();
    assert(size >= num_folds);
    for (size_t i = 0; i < num_folds; ++i) {
      size_t begin = i * size / num_folds;
      size_t end = (i + 1) * size / num_folds;
      std::vector<slice> train_slices;
      train_slices.push_back(slice(0, begin));
      train_slices.push_back(slice(end, size));
      train.push_back(ds.subset(train_slices));
      test.push_back(ds.subset(begin, end));
    }
  }

} // namespace sill

#endif

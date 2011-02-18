
#include <sill/learning/dataset/dataset_statistics.hpp>
#include <sill/learning/learn_factor.hpp>

#include <sill/macros_def.hpp>

namespace sill {

    // Public methods: order statistics
    //==========================================================================

    void dataset_statistics::compute_order_stats(size_t v) {
      if (ds.empty())
        return;
      if (order_stats_.size() != ds.vector_dim())
        order_stats_.resize(ds.vector_dim());
      if (order_stats_[v].size() == ds.size())
        return;
      order_stats_[v].resize(ds.size());
      for (size_t i = 0; i < ds.size(); i++)
        order_stats_[v][i] = i;
      std::sort(order_stats_[v].begin(), order_stats_[v].end(),
                order_stats_functor(ds, v));
    }

    void dataset_statistics::compute_order_stats() {
      if (ds.empty())
        return;
      for (size_t v = 0; v < ds.vector_dim(); v++)
        compute_order_stats(v);
    }

    // Public methods: mutual information
    //==========================================================================

    void dataset_statistics::compute_mutual_info
    (size_t i, size_t j, double smoothing) {
      if (smoothing < 0)
        smoothing = .01 / ds.size();
      assert(i != j);
      if (i > j) {
        size_t k = i;
        i = j;
        j = k;
      }
      if (ds.empty())
        return;
      if (mutual_info_.size() != ds.num_finite() - 1)
        mutual_info_.resize(ds.num_finite() - 1);
      if (mutual_info_[i].size() == ds.num_finite() - i - 1)
        mutual_info_[i].resize(ds.num_finite() - i - 1);
      assert(i < ds.num_finite() && j < ds.num_finite());
      finite_variable* vi = ds.finite_list()[i];
      finite_variable* vj = ds.finite_list()[j];
      table_factor fij
        (learn_marginal<table_factor>(make_domain<finite_variable>(vi, vj), ds,
                                      smoothing));
      mutual_info_[i][j - i - 1] =
        fij.mutual_information(make_domain(vi), make_domain(vj));
    }

    void dataset_statistics::compute_mutual_info(double smoothing) {
      if (ds.empty())
        return;
      for (size_t i = 0; i < ds.num_finite() - 1; ++i)
        for (size_t j = i+1; j < ds.num_finite(); ++j)
          compute_mutual_info(i, j, smoothing);
    }

  // Public methods: marginals
  //==========================================================================

  const table_factor&
  dataset_statistics::marginal(const finite_domain& d, double smoothing) const {
    if (!(marginal_map.count(d)))
      marginal_map[d] = learn_marginal<table_factor>(d, ds, smoothing);
    return marginal_map[d];
  }

  void dataset_statistics::clear_marginals() const {
    marginal_map.clear();
  }

  // Public methods: min/max of each vector variable
  //==========================================================================

  vec
  dataset_statistics::vector_var_min(const dataset& data,
                             const vector_var_vector& vars){
    vec mins(vector_size(vars));
    vec vals(mins.size());
    foreach(const record& r, data.records()) {
      r.vector_values(vals, vars);
      for (size_t j(0); j < vals.size(); ++j) {
        if (mins[j] > vals[j])
          mins[j] = vals[j];
      }
    }
    return mins;
  }

  vec
  dataset_statistics::vector_var_max(const dataset& data,
                             const vector_var_vector& vars){
    vec maxs(vector_size(vars));
    vec vals(maxs.size());
    foreach(const record& r, data.records()) {
      r.vector_values(vals, vars);
      for (size_t j(0); j < vals.size(); ++j) {
        if (maxs[j] < vals[j])
          maxs[j] = vals[j];
      }
    }
    return maxs;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

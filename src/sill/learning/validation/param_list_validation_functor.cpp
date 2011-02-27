
#include <sill/learning/validation/param_list_validation_functor.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  param_list_validation_functor::param_list_validation_functor()
    : unif_int(0, std::numeric_limits<int>::max()) { }

  void
  param_list_validation_functor::test(const std::vector<vec>& lambdas,
                                      const dataset& train_ds,
                                      const dataset& test_ds,
                                      unsigned random_seed,
                                      model_validation_functor& mv_func) {
    typedef std::pair<std::string, double> result_value_pair;

    std::vector<size_t> sorted_lambda_indices(sorted_indices(lambdas));
    results.resize(lambdas.size());
    results.zeros_memset();
    all_results.clear();
    boost::mt11213b rng(random_seed);

    for (size_t i_ = 0; i_ < lambdas.size(); ++i_) {
      size_t i = sorted_lambda_indices[(lambdas.size() - 1) - i_];
      if (i_ == 0) {
        results[i] = mv_func.train(train_ds, test_ds, lambdas[i], false,
                                   unif_int(rng));
        foreach(const result_value_pair& rvp, mv_func.result_map()) {
          all_results[rvp.first] = vec(lambdas.size(), 0);
          all_results[rvp.first][i] = rvp.second;
        }
      } else {
        results[i] = mv_func.train(train_ds, test_ds, lambdas[i], true,
                                   unif_int(rng));
        foreach(const result_value_pair& rvp, mv_func.result_map()) {
          all_results[rvp.first][i] = rvp.second;
        }
      }
    }
  } // test

} // namespace sill

#include <sill/macros_undef.hpp>

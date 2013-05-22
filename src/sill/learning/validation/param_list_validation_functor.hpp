#ifndef SILL_PARAM_LIST_VALIDATION_FUNCTOR_HPP
#define SILL_PARAM_LIST_VALIDATION_FUNCTOR_HPP

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

#include <sill/learning/validation/crossval_parameters.hpp>
#include <sill/learning/validation/model_validation_functor.hpp>
#include <sill/learning/validation/parameter_grid.hpp>
#include <sill/math/statistics.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Functor which calls a model_validation_functor on a list of parameter
   * values.
   *
   * By default, this functor handles the parameters in decreasing order,
   * which helps facilitate warm starts to speed up optimization.
   * However, the test() method may be overridden if necessary.
   *
   * @tparam LA  Linear algebra type specifier
   *             (default = dense_linear_algebra<>)
   */
  template <typename LA = dense_linear_algebra<> >
  struct param_list_validation_functor {

    typedef LA la_type;

    typedef typename la_type::vector_type vector_type;
    typedef typename la_type::value_type  value_type;

    //! Results corresponding to the last parameters passed to test().
    //! These results are the ones used to choose the best parameters.
    vec results;

    //! Results corresponding to the last lambdas passed to test().
    //! These results are all ones returned by the model_validation_functor.
    std::map<std::string, vec> all_results;

    param_list_validation_functor()
      : unif_int(0, std::numeric_limits<int>::max()) { }

    virtual ~param_list_validation_functor() { }

    /**
     * Calls model_validation_functor on a list of parameter values.
     *
     * This tests the parameters in decreasing order,
     * which helps facilitate warm starts to speed up optimization.
     * If parameter vecs have length > 1, then they are lexigraphically sorted.
     * Override this method if necessary.
     *
     * @param lambdas  List of parameter values to test.
     */
    virtual void test(const std::vector<vec>& lambdas,
                      const dataset<la_type>& train_ds,
                      const dataset<la_type>& test_ds,
                      unsigned random_seed,
                      model_validation_functor<la_type>& mv_func) {
      typedef std::pair<std::string, value_type> result_value_pair;

      std::vector<size_t> sorted_lambda_indices(sorted_indices(lambdas));
      results.zeros(lambdas.size());
      all_results.clear();
      boost::mt11213b rng(random_seed);

      for (size_t i_ = 0; i_ < lambdas.size(); ++i_) {
        size_t i = sorted_lambda_indices[(lambdas.size() - 1) - i_];
        if (i_ == 0) {
          results[i] = mv_func.train(train_ds, test_ds, lambdas[i], false,
                                     unif_int(rng));
          foreach(const result_value_pair& rvp, mv_func.result_map()) {
            all_results[rvp.first] = zeros<vec>(lambdas.size());
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

  protected:

    boost::uniform_int<int> unif_int;

  }; // struct param_list_validation_functor

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_PARAM_LIST_VALIDATION_FUNCTOR_HPP

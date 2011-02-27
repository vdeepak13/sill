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
   */
  struct param_list_validation_functor {

    //! Results corresponding to the last parameters passed to test().
    //! These results are the ones used to choose the best parameters.
    vec results;

    //! Results corresponding to the last lambdas passed to test().
    //! These results are all ones returned by the model_validation_functor.
    std::map<std::string, vec> all_results;

    param_list_validation_functor();

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
                      const dataset& train_ds,
                      const dataset& test_ds,
                      unsigned random_seed,
                      model_validation_functor& mv_func);

  protected:

    boost::uniform_int<int> unif_int;

  }; // struct param_list_validation_functor

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_PARAM_LIST_VALIDATION_FUNCTOR_HPP

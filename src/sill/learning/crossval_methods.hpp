
#ifndef SILL_CROSSVAL_METHODS_HPP
#define SILL_CROSSVAL_METHODS_HPP

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

#include <sill/learning/validation/crossval_parameters.hpp>
#include <sill/learning/dataset/dataset.hpp>
#include <sill/learning/validation/parameter_grid.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Template for iteratively running cross validation to choose a parameter
   * vector lambda via "lambda zooming" as follows:
   *  - Choose a grid of lambda values, as specified by cv_params.
   *  - Do CV to find the best point in the grid.
   *  - Choose a new grid of lambda values around the best point,
   *    and test those.
   *  - Iterate for cv_params.zoom iterations.
   * If the problem of choosing lambda is convex, then this is an efficient
   * way of trying out finer-grained lambdas without testing an enormous grid.
   *
   * @param reg_params  (Return value.) lambdas which were tried
   * @param means       (Return value.) means[i] = avg score for lambdas[i]
   *                    NOTE: This tries to MINIMIZE this score.
   * @param stderrs     (Return value.) corresponding std errors of scores
   * @param cv_params   Parameters specifying how to do cross validation.
   * @param cv_functor  Functor which does CV for a fixed set of lambdas and
   *                    returns the best lambda found.
   * @param random_seed This uses this random seed, not the one in the
   *                    algorithm parameters.
   *
   * @return  chosen lambda
   *
   * @tparam CrossvalFunctor  Functor which does CV for a fixed set of lambdas
   *                          and returns the best lambda found.
   * @tparam N                dimensionality of lambda vector
   */
  template <typename CrossvalFunctor>
  vec crossval_zoom(std::vector<vec>& lambdas, vec& means, vec& stderrs,
                    const crossval_parameters& cv_params,
                    const CrossvalFunctor& cv_functor, unsigned random_seed) {
    assert(cv_params.valid());

    boost::mt11213b rng(random_seed);
    boost::uniform_int<int> unif_int(0, std::numeric_limits<int>::max());
    lambdas.resize(0);
    means.set_size(0);
    stderrs.set_size(0);

    // These hold values for each round of zooming:
    std::vector<vec>
      lambdas_zoom(create_parameter_grid(cv_params.minvals, cv_params.maxvals,
                                         cv_params.nvals, cv_params.log_scale));
    vec means_zoom(lambdas_zoom.size(), 0);
    vec stderrs_zoom(lambdas_zoom.size(), 0);

    size_t best_i(0); // This indexes the current best value in lambdas.

    for (size_t zoom_i(0); zoom_i <= cv_params.zoom; ++zoom_i) {
      if (zoom_i > 0) {
        lambdas_zoom =
          zoom_parameter_grid(lambdas, lambdas[best_i], cv_params.nvals,
                              cv_params.log_scale);
        means_zoom.zeros(lambdas_zoom.size());
        stderrs_zoom.zeros(lambdas_zoom.size());
      }
      vec best_lambda(cv_functor(means_zoom, stderrs_zoom, lambdas_zoom,
                                 cv_params.nfolds, unif_int(rng)));
      size_t best_zoom_i(0);
      for (size_t j(0); j < lambdas_zoom.size(); ++j) {
        if (equal(lambdas_zoom[j], best_lambda)) {
          best_zoom_i = j;
          break;
        }
      }
      if (means.size() == 0) {
        best_i = best_zoom_i;
      } else  if (means[best_i] > means_zoom[best_zoom_i]) {
        best_i = means.size() + best_zoom_i;
      }
      size_t oldsize(lambdas.size());
      lambdas.resize(oldsize + lambdas_zoom.size());
      means.reshape(oldsize + means_zoom.size(), 1);
      stderrs.reshape(oldsize + stderrs_zoom.size(), 1);
      for (size_t j(0); j < lambdas_zoom.size(); ++j) {
        lambdas[oldsize + j] = lambdas_zoom[j];
        means[oldsize + j] = means_zoom[j];
        stderrs[oldsize + j] = stderrs_zoom[j];
      }
    }
    return lambdas[best_i];
  } // crossval_zoom

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_CROSSVAL_METHODS_HPP

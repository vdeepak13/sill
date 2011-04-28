#ifndef SILL_CRF_VALIDATION_FUNCTOR_HPP
#define SILL_CRF_VALIDATION_FUNCTOR_HPP

#include <sill/learning/crf/crf_parameter_learner.hpp>
#include <sill/learning/validation/model_validation_functor.hpp>
#include <sill/model/crf_model.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Forward declarations
  struct crf_parameter_learner_parameters;
  template <typename F> class crf_parameter_learner;

  /**
   * Model validation functor for Conditional Random Fields (CRFs).
   *
   * @tparam F  CRF factor type.
   */
  template <typename F>
  class crf_validation_functor
    : public model_validation_functor<> {

    // Public types
    // =========================================================================
  public:

    typedef model_validation_functor<> base;

    //! Type of CRF factor
    typedef F crf_factor_type;

    //! Type of crf_graph.
    typedef typename crf_model<F>::crf_graph_type crf_graph_type;

    // Constructors and destructors
    // =========================================================================

    /**
     * Constructor which uses a crf_graph.
     * WARNING: This does not work with templated factors; use the below
     *          constructor instead.
     */
    crf_validation_functor(const crf_graph_type& structure,
                           const crf_parameter_learner_parameters& cpl_params)
      : structure(structure), model(structure), cpl_params(cpl_params) {
      this->use_weights = false;
    }

    /**
     * Constructor which uses a crf_model.
     *
     * @param use_weights   If true, then use the given model's weights
     *                      (parameters) to initialize learning.
     */
    crf_validation_functor(const crf_model<F>& model, bool use_weights,
                           const crf_parameter_learner_parameters& cpl_params)
      : structure(model), model(model), cpl_params(cpl_params) {
      this->use_weights = use_weights;
    }

    // Protected data
    // =========================================================================
  protected:

    const crf_graph_type& structure;

    crf_model<F> model;

    crf_parameter_learner_parameters cpl_params;

    // Protected methods
    // =========================================================================

    void train_model(const dataset<>& ds, unsigned random_seed) {
      cpl_params.random_seed = random_seed;
      if (model.num_arguments() != 0) {
        crf_parameter_learner<F> cpl(model, !use_weights, ds, cpl_params);
        model = cpl.current_model();
      } else {
        assert(false); // This version does not work with templated factors. Figure out a way to resolve this issue.
        crf_parameter_learner<F> cpl(structure, ds, cpl_params);
        model = cpl.current_model();
      }
    }

    void train_model(const dataset<>& ds, const vector_type& validation_params,
                     unsigned random_seed) {
      cpl_params.lambdas = validation_params;
      train_model(ds, random_seed);
    }

    double add_results(const dataset<>& ds, const std::string& prefix) {
      double ll = model.expected_log_likelihood(ds);
      result_map_[prefix + "log likelihood"] = ll;
      result_map_[prefix + "per-label accuracy"] =
        model.expected_per_label_accuracy(ds);
      return ll;
    }

  }; // class crf_validation_functor<F>

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_CRF_VALIDATION_FUNCTOR_HPP

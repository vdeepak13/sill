#ifndef SILL_EVALUATE_MODEL_HPP
#define SILL_EVALUATE_MODEL_HPP

#include <sill/learning/crf/crf_X_mapping.hpp>
#include <sill/learning/dataset_old/dataset.hpp>
#include <sill/learning/discriminative/multiclass_classifier.hpp>
#include <sill/math/statistics.hpp>
#include <sill/model/crf_model.hpp>
#include <sill/model/decomposable.hpp>

#include <sill/macros_def.hpp>

/**
 * \file evaluate_model.hpp  Set of methods for analyzing graphical models.
 *
 * This includes:
 *  - Evaluating the performance of graphical models on data sets.
 */

namespace sill {

  /**
   * Compute the precision and recall of a model on a dataset.
   *  - precision = (# correctly predicted positives) / (# predicted positives)
   *  - recall = (# correctly predicted positives) / (# actual positives)
   * The counts are computed via sums over both samples and labels.
   *
   * @param crf  Model whose output variables Y must all be binary-valued.
   *
   * @return <precision,recall>
   */
  template <typename F, typename LA>
  std::pair<double,double>
  precision_recall(const crf_model<F>& crf, const dataset<LA>& ds) {
    size_t n_correct_pred = 0;
    size_t n_pred = 0;
    size_t actual = 0;
    foreach(typename F::output_variable_type* v, crf.output_arguments()) {
      assert(v->type() == variable::FINITE_VARIABLE);
      finite_variable* fv = static_cast<finite_variable*>(v);
      assert(fv->size() == 2);
    }
    foreach(const record<LA>& r, ds.records()) {
      const decomposable<typename F::output_factor_type>& Ygivenx =
        crf.condition(r);
      finite_assignment mpa(Ygivenx.max_prob_assignment());
      foreach(finite_variable* v, Ygivenx.arguments()) {
        if (r.finite(v) == 1) {
          ++actual;
          if (mpa[v] == 1)
            ++n_correct_pred;
        }
        if (mpa[v] == 1)
          ++n_pred;
      }
    }
    double precision = (n_correct_pred + 0.) / n_pred;
    double recall = (n_correct_pred + 0.) / actual;
    return std::make_pair(precision, recall);
  } // precision_recall

  /**
   * Compute the precision and recall of a model on a dataset.
   *  - precision = (# correctly predicted positives) / (# predicted positives)
   *  - recall = (# correctly predicted positives) / (# actual positives)
   * The counts are computed via sums over both samples and labels.
   *
   * @return <precision,recall>
   */
  template <typename LA1, typename LA2>
  std::pair<double,double>
  precision_recall(const multiclass_classifier<LA1>& model,
                   const dataset<LA2>& ds) {
    assert(ds.has_variable(model.label()));
    size_t n_correct_pred = 0;
    size_t n_pred = ds.size();
    size_t actual = ds.size();
    foreach(const record<LA2>& r, ds.records()) {
      if (r.finite(model.label()) == model.predict(r))
        ++n_correct_pred;
    }
    double precision = (n_correct_pred + 0.) / n_pred;
    double recall = (n_correct_pred + 0.) / actual;
    return std::make_pair(precision, recall);
  } // precision_recall

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

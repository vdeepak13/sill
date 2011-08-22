
#ifndef SILL_EVALUATE_MODEL_HPP
#define SILL_EVALUATE_MODEL_HPP

#include <sill/learning/crf/crf_X_mapping.hpp>
#include <sill/learning/dataset/dataset.hpp>
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
 *  - Evaluating how well a distribution obeys certain properties.
 *     - Score Decay Assumption
 *
 * @todo Move the SDA part elsewhere.
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
      assert(v->get_variable_type() == variable::FINITE_VARIABLE);
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

  /**
   * Class for evaluating how well a distribution represented by P(X), P(Y|X)
   * obeys the Score Decay Assumption (SDA) w.r.t. certain scores.
   *
   * For details on the SDA, see Bradley and Guestrin. "Learning Tree
   *  Conditional Random Fields." ICML 2010.
   * This measures the SDA w.r.t. the scores used by the pwl_crf_learner class.
   * For these scores:
   *  - PWL
   *  - (local) CMI
   *  - DCI
   * Compute these measures:
   *  - For all:
   *     - triplets (i,j,k) on a single path in the tree
   *        (SDA)
   *     - consecutive triplets (i,j,k)
   *        (approxSDA)
   *  - Compute:
   *     - SDA violation score(i,k) - (1/2)(score(i,j) + score(j,k))
   *        (violation)
   *     - binary SDA violation (1/2)(I[score(i,k) >= score(i,j)] +
   *                                  I[score(i,k) >= score(j,k)])
   *        (indicator)
   * Compute the mean/stderr and median/MAD of these measures.
   *
   * @tparam FactorType    Type of factor for model of P(Y,X).
   * @tparam CRFfactor     Type of factor used for P(Y|X).
   *                       TO DO: Is this necessary?
   */
  template <typename FactorType, typename CRFfactor>
  struct evaluate_SDA_on_CRF {

    typedef typename FactorType::variable_type variable_type;

    typedef typename FactorType::domain_type domain_type;

    typedef typename decomposable<FactorType>::vertex vertex;

    // Private types and data members
    //--------------------------------------------------------------------------
  private:

    /**
     * Functor used to compute SDA violation measures for all triplets (i,j,k)
     * on paths in the tree.
     */
    struct evaluate_triplet_functor {
    private:
      const mat* pwl_scores;
      const mat* cmi_scores;
      const mat* dci_scores;
      const std::map<variable_type*, size_t>* vertex_map;

      std::vector<double>* pwl_SDA_violations;
      std::vector<double>* cmi_SDA_violations;
      std::vector<double>* dci_SDA_violations;
      std::vector<double>* pwl_SDA_indicators;
      std::vector<double>* cmi_SDA_indicators;
      std::vector<double>* dci_SDA_indicators;

    public:
      evaluate_triplet_functor
      (const mat& pwl_scores, const mat& cmi_scores, const mat& dci_scores,
       const std::map<variable_type*, size_t>& vertex_map,
       std::vector<double>& pwl_SDA_violations,
       std::vector<double>& cmi_SDA_violations,
       std::vector<double>& dci_SDA_violations,
       std::vector<double>& pwl_SDA_indicators,
       std::vector<double>& cmi_SDA_indicators,
       std::vector<double>& dci_SDA_indicators)
        : pwl_scores(&pwl_scores), cmi_scores(&cmi_scores),
          dci_scores(&dci_scores), vertex_map(&vertex_map),
          pwl_SDA_violations(&pwl_SDA_violations),
          cmi_SDA_violations(&cmi_SDA_violations),
          dci_SDA_violations(&dci_SDA_violations),
          pwl_SDA_indicators(&pwl_SDA_indicators),
          cmi_SDA_indicators(&cmi_SDA_indicators),
          dci_SDA_indicators(&dci_SDA_indicators) { }

      void
      operator()(variable_type* v_i, variable_type* v_j, variable_type* v_k) {
        size_t i(safe_get(*vertex_map, v_i));
        size_t j(safe_get(*vertex_map, v_j));
        size_t k(safe_get(*vertex_map, v_k));
        if (i > j)
          std::swap(i,j);
        if (j > k)
          std::swap(j,k);
        if (i > j)
          std::swap(i,j);
        double val1;
        double val2;
        val1 = pwl_scores->operator()(i,k) - pwl_scores->operator()(i,j);
        val2 = pwl_scores->operator()(i,k) - pwl_scores->operator()(j,k);
        pwl_SDA_violations->push_back((val1 + val2) / 2.);
        pwl_SDA_indicators->push_back((val1 >= 0 ? .5 : 0) +
                                      (val2 >= 0 ? .5 : 0));
        val1 = cmi_scores->operator()(i,k) - cmi_scores->operator()(i,j);
        val2 = cmi_scores->operator()(i,k) - cmi_scores->operator()(j,k);
        cmi_SDA_violations->push_back((val1 + val2) / 2.);
        cmi_SDA_indicators->push_back((val1 >= 0 ? .5 : 0) +
                                      (val2 >= 0 ? .5 : 0));
        val1 = dci_scores->operator()(i,k) - dci_scores->operator()(i,j);
        val2 = dci_scores->operator()(i,k) - dci_scores->operator()(j,k);
        dci_SDA_violations->push_back((val1 + val2) / 2.);
        dci_SDA_indicators->push_back((val1 >= 0 ? .5 : 0) +
                                      (val2 >= 0 ? .5 : 0));
      }

    }; // struct evaluate_triplet_functor

  public:
    // Evaluation statistics (See descriptions above.)
    //--------------------------------------------------------------------------

    std::pair<double, double> mean_pwl_SDA_violation;
    std::pair<double, double> median_pwl_SDA_violation;

    std::pair<double, double> mean_cmi_SDA_violation;
    std::pair<double, double> median_cmi_SDA_violation;

    std::pair<double, double> mean_dci_SDA_violation;
    std::pair<double, double> median_dci_SDA_violation;

    std::pair<double, double> mean_pwl_SDA_indicator;
    std::pair<double, double> median_pwl_SDA_indicator;

    std::pair<double, double> mean_cmi_SDA_indicator;
    std::pair<double, double> median_cmi_SDA_indicator;

    std::pair<double, double> mean_dci_SDA_indicator;
    std::pair<double, double> median_dci_SDA_indicator;

    std::pair<double, double> mean_pwl_approxSDA_violation;
    std::pair<double, double> median_pwl_approxSDA_violation;

    std::pair<double, double> mean_cmi_approxSDA_violation;
    std::pair<double, double> median_cmi_approxSDA_violation;

    std::pair<double, double> mean_dci_approxSDA_violation;
    std::pair<double, double> median_dci_approxSDA_violation;

    std::pair<double, double> mean_pwl_approxSDA_indicator;
    std::pair<double, double> median_pwl_approxSDA_indicator;

    std::pair<double, double> mean_cmi_approxSDA_indicator;
    std::pair<double, double> median_cmi_approxSDA_indicator;

    std::pair<double, double> mean_dci_approxSDA_indicator;
    std::pair<double, double> median_dci_approxSDA_indicator;

    //! pwl_scores(i,j) = score for edge (i,j) where i < j
    mat pwl_scores;

    //! cmi_scores(i,j) = score for edge (i,j) where i < j
    mat cmi_scores;

    //! dci_scores(i,j) = score for edge (i,j) where i < j
    mat dci_scores;

    // Public methods
    //==========================================================================

    //! Constructor which does nothing; call evaluate() to compute results.
    evaluate_SDA_on_CRF() {
      clear();
    }

    /**
     * Constructor which calls evaluate(); see evaluate() for details.
     */
    evaluate_SDA_on_CRF
    (const decomposable<FactorType>& model,
     const undirected_graph<variable_type*>& structure,
     const std::vector<variable_type>& Yvars,
     const crf_X_mapping<CRFfactor>& Xmap,
     double mult_std_error, unsigned random_seed) {
      clear();
      evaluate(model, structure, Yvars, Xmap, mult_std_error, random_seed);
    }

    /**
     * This evaluates the given model P(Y,X), using approximate entropy
     * computation from samples.
     *
     * @param model     P(Y,X). This should be normalized already.
     * @param structure Graph over Y showing the tree structure of the model.
     * @param Xmap      PWL CRF learner X mapping.
     * @param mult_std_error  Convergence criterion for entropy estimates;
     *                        see decomposable::approx_conditional_entropy().
     * @param approx_only  If true, only compute the approxSDA violations.
     *                     (default = false)
     */
    void evaluate
    (const decomposable<FactorType>& model,
     const undirected_graph<variable_type*>& structure,
     const std::vector<variable_type*>& Yvars,
     const crf_X_mapping<CRFfactor>& Xmap,
     double mult_std_error, unsigned random_seed, bool approx_only = false) {

      std::cerr << "SDA evaluate() method.\n"
                << "   Computing scores..." << std::endl;

      size_t n(Yvars.size());
      if (n <= 2)
        return;
      assert(mult_std_error > 0);

      boost::mt11213b rng(random_seed);
      std::map<variable_type*, size_t> vertex_map;

      // Compute *_scores
      pwl_scores.set_size(n-1, n);
      cmi_scores.set_size(n-1, n);
      dci_scores.set_size(n-1, n);
      vec H_Yi_xi(n); // H(Yi | Xi)
      for (size_t i(0); i < Yvars.size(); ++i) {
        H_Yi_xi[i] =
          model.approx_conditional_entropy
          (make_domain(Yvars[i]), *(Xmap[make_domain(Yvars[i])]),
           mult_std_error, rng).first;
        vertex_map[Yvars[i]] = i;
      }
      for (size_t i(0); i < n - 1; ++i) {
        for (size_t j(i+1); j < n; ++j) {
          double H_Yi_xij = 
            model.approx_conditional_entropy
            (make_domain(Yvars[i]), *(Xmap[make_domain(Yvars[i],Yvars[j])]),
             mult_std_error, rng).first;
          double H_Yj_xij = 
            model.approx_conditional_entropy
            (make_domain(Yvars[j]), *(Xmap[make_domain(Yvars[i],Yvars[j])]),
             mult_std_error, rng).first;
          double H_Yij_xij = 
            model.approx_conditional_entropy
            (make_domain(Yvars[i],Yvars[j]),
             *(Xmap[make_domain(Yvars[i],Yvars[j])]),
             mult_std_error, rng).first;
          pwl_scores(i,j) = - H_Yij_xij;
          cmi_scores(i,j) = H_Yi_xij + H_Yj_xij - H_Yij_xij;
          dci_scores(i,j) = H_Yi_xi[i] + H_Yi_xi[j] - H_Yij_xij;
        }
      }

      std::vector<double> pwl_SDA_violations;
      std::vector<double> cmi_SDA_violations;
      std::vector<double> dci_SDA_violations;
      std::vector<double> pwl_SDA_indicators;
      std::vector<double> cmi_SDA_indicators;
      std::vector<double> dci_SDA_indicators;
      evaluate_triplet_functor
        et_functor(pwl_scores, cmi_scores, dci_scores, vertex_map,
                   pwl_SDA_violations, cmi_SDA_violations, dci_SDA_violations,
                   pwl_SDA_indicators, cmi_SDA_indicators, dci_SDA_indicators);

      if (!approx_only) {
        std::cerr << "  Computing SDA violations for all triplets..."
                  << std::endl;

        // Compute SDA violation measures for all triplets on paths.
        visit_triplets_on_paths(structure, et_functor);

        mean_pwl_SDA_violation = mean_stderr(pwl_SDA_violations);
        median_pwl_SDA_violation = median_MAD(pwl_SDA_violations);
        mean_cmi_SDA_violation = mean_stderr(cmi_SDA_violations);
        median_cmi_SDA_violation = median_MAD(cmi_SDA_violations);
        mean_dci_SDA_violation = mean_stderr(dci_SDA_violations);
        median_dci_SDA_violation = median_MAD(dci_SDA_violations);
        mean_pwl_SDA_indicator = mean_stderr(pwl_SDA_indicators);
        median_pwl_SDA_indicator = median_MAD(pwl_SDA_indicators);
        mean_cmi_SDA_indicator = mean_stderr(cmi_SDA_indicators);
        median_cmi_SDA_indicator = median_MAD(cmi_SDA_indicators);
        mean_dci_SDA_indicator = mean_stderr(dci_SDA_indicators);
        median_dci_SDA_indicator = median_MAD(dci_SDA_indicators);

        pwl_SDA_violations.clear();
        cmi_SDA_violations.clear();
        dci_SDA_violations.clear();
        pwl_SDA_indicators.clear();
        cmi_SDA_indicators.clear();
        dci_SDA_indicators.clear();
      }

      // Compute SDA violation measures for all consecutive triplets.
      std::cerr << "  Computing SDA violations for consecutive triplets..."
                << std::endl;
      visit_consecutive_triplets(structure, et_functor);

      mean_pwl_approxSDA_violation = mean_stderr(pwl_SDA_violations);
      median_pwl_approxSDA_violation =median_MAD(pwl_SDA_violations);
      mean_cmi_approxSDA_violation = mean_stderr(cmi_SDA_violations);
      median_cmi_approxSDA_violation =median_MAD(cmi_SDA_violations);
      mean_dci_approxSDA_violation = mean_stderr(dci_SDA_violations);
      median_dci_approxSDA_violation =median_MAD(dci_SDA_violations);
      mean_pwl_approxSDA_indicator = mean_stderr(pwl_SDA_indicators);
      median_pwl_approxSDA_indicator =median_MAD(pwl_SDA_indicators);
      mean_cmi_approxSDA_indicator = mean_stderr(cmi_SDA_indicators);
      median_cmi_approxSDA_indicator =median_MAD(cmi_SDA_indicators);
      mean_dci_approxSDA_indicator = mean_stderr(dci_SDA_indicators);
      median_dci_approxSDA_indicator =median_MAD(dci_SDA_indicators);
    } // end of constructor evaluate_SDA_on_CRF

    //! Clears all results.
    void clear() {
      mean_pwl_SDA_violation = std::make_pair(0,0);
      median_pwl_SDA_violation = std::make_pair(0,0);
      mean_cmi_SDA_violation = std::make_pair(0,0);
      median_cmi_SDA_violation = std::make_pair(0,0);
      mean_dci_SDA_violation = std::make_pair(0,0);
      median_dci_SDA_violation = std::make_pair(0,0);
      mean_pwl_SDA_indicator = std::make_pair(0,0);
      median_pwl_SDA_indicator = std::make_pair(0,0);
      mean_cmi_SDA_indicator = std::make_pair(0,0);
      median_cmi_SDA_indicator = std::make_pair(0,0);
      mean_dci_SDA_indicator = std::make_pair(0,0);
      median_dci_SDA_indicator = std::make_pair(0,0);
      mean_pwl_approxSDA_violation = std::make_pair(0,0);
      median_pwl_approxSDA_violation = std::make_pair(0,0);
      mean_cmi_approxSDA_violation = std::make_pair(0,0);
      median_cmi_approxSDA_violation = std::make_pair(0,0);
      mean_dci_approxSDA_violation = std::make_pair(0,0);
      median_dci_approxSDA_violation = std::make_pair(0,0);
      mean_pwl_approxSDA_indicator = std::make_pair(0,0);
      median_pwl_approxSDA_indicator = std::make_pair(0,0);
      mean_cmi_approxSDA_indicator = std::make_pair(0,0);
      median_cmi_approxSDA_indicator = std::make_pair(0,0);
      mean_dci_approxSDA_indicator = std::make_pair(0,0);
      median_dci_approxSDA_indicator = std::make_pair(0,0);
    }

  }; // struct evaluate_SDA_on_CRF

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

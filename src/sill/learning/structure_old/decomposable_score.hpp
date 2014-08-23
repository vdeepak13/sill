#ifndef SILL_LEARNING_DECOMPOSABLE_SCORE_HPP
#define SILL_LEARNING_DECOMPOSABLE_SCORE_HPP

#include <sill/learning/structure_old/decomposable_change.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Virtual class for computing a score for score-based structure search over
   * decomposable models.  The score may be either:
   *  - global: It must be recomputed after every change, though it may permit
   *            fast estimates.
   *  - local: It can be recomputed exactly after a change using minimal
   *           computation.
   *
   * @param F               type of factor used in the decomposable model
   *
   * \author Joseph Bradley
   * \ingroup learning_structure
   */
  template <typename F>
  class decomposable_score {

  public:
    //! The type of factor used in the decomposable model
    typedef F factor_type;

    //! The type of variable associated with a factor
    //    typedef typename F::variable_type variable_type;

    //! The domain type of the factor
    //    typedef typename F::domain_type domain_type;

    //    //! The type of edge associated with the model
    //    typedef typename decomposable<F>::edge edge;

    //    //! the type of vertex associated with the model
    //    typedef typename decomposable<F>::vertex vertex;

    //! Indicator for global/local score
    const bool is_global;

    //! Indicates whether this score can be estimated significantly faster
    //! than it can be computed exactly
    const bool has_estimate;

    //////////////////////// PUBLIC METHODS ///////////////////////////////

    decomposable_score(bool is_global, bool has_estimate)
      : is_global(is_global), has_estimate(has_estimate) {
    }

    virtual ~decomposable_score() { }

    //! Given a model, compute the score (from scratch)
    virtual double compute(const learnt_decomposable<F>& model) const = 0;

    //! Given a model, current score, and change to the model (not yet made),
    //! compute the change in the score.
    //! This takes advantage of local scores when possible.
    //! @return <true if move is valid, [new score - cur_score]>
    virtual std::pair<bool, double>
    compute_change(const learnt_decomposable<F>& model, double cur_score,
                   const decomposable_change<F>& change,
                   dataset_statistics<>& stats) const {
      if (is_global) {
        if (!(change.valid(model)))
          return std::make_pair(false, 0.);
        learnt_decomposable<F> tmpmodel(model);
        change.commit(tmpmodel, stats);
        return std::make_pair(true, (compute(tmpmodel) - cur_score));
      } else
        assert(false); // This should be overridden when the score is local.
    }

    //! Given a model, compute an estimate of the score.
    //! This computes the exact score if the score cannot be estimated.
    virtual double estimate(const learnt_decomposable<F>& model) const = 0;

    //! Given a model, current score, and change to the model (not yet made),
    //! estimate the change in the score.
    //! If the score is local, then this computes the exact score change.
    //! If the score is global, then this computes an estimate if possible
    //! and an exact score change otherwise.
    //! @return <true if move is valid, [new score - cur_score]>
    virtual std::pair<bool, double>
    estimate_change(const learnt_decomposable<F>& model, double cur_score,
                    const decomposable_change<F>& change,
                    dataset_statistics<>& stats) const {
      if (is_global && has_estimate)
          assert(false); // This should be overridden by an estimate.
      else
        return compute_change(model, cur_score, change, stats);
    }

  }; // class decomposable_score

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DECOMPOSABLE_SCORE_HPP

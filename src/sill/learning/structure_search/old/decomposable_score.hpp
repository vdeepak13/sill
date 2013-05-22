
#ifndef SILL_LEARNING_DECOMPOSABLE_SCORE_HPP
#define SILL_LEARNING_DECOMPOSABLE_SCORE_HPP

#include <sill/learning/structure_learning/decomposable_change.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Class for computing a score for score-based structure search over
   * decomposable models.  The score may be either:
   *  - global: It must be recomputed after every change, though it may permit
   *            fast estimates.
   *  - local: It can be recomputed exactly after a change using minimal
   *           computation.
   *
   * @param F            type of factor used in the decomposable model
   * @param MarginalPtr  MarginalPtr used by decomposable_change
   *
   * \author Joseph Bradley
   * \ingroup learning_structure
   */
  template <typename F, typename MarginalPtr = boost::shared_ptr<F> >
  class decomposable_score {

  public:
    //! The type of factor used in the decomposable model
    typedef F factor_type;

    //! The type of variable associated with a factor
    //    typedef typename F::variable_type variable_type;

    //! The domain type of the factor
    //    typedef typename F::domain_type domain_type;

    //! The type of edge associated with the model
    typedef typename decomposable<F>::edge edge;

    //! the type of vertex associated with the model
    typedef typename decomposable<F>::vertex vertex;

    virtual ~decomposable_score() { }

    //! Indicator for global/local score
    static const bool is_global;

    //! Indicates whether this score can be estimated significantly faster
    //! than it can be computed exactly
    static const bool has_estimate;

    //! Given a model, compute the score (from scratch)
    static double compute(const decomposable<F>& model) = 0;

    //! Given a model, current score, and change to the model (not yet made),
    //! compute the change in the score
    virtual static double
    compute_change(const decomposable<F>& model, double cur_score,
                   const decomposable_change<F,MarginalPtr>& change) = 0;

    //! Given a model, compute an estimate of the score.
    //! This is only valid if has_estimate == true.
    virtual static double estimate(const decomposable<F>& model) = 0;

    //! Given a model, current score, and change to the model (not yet made),
    //! estimate the change in the score.
    //! This is only valid if has_estimate == true.
    virtual static double
    estimate_change(const decomposable<F>& model, double cur_score,
                    const decomposable_change<F,MarginalPtr>& change) = 0;

  }; // class decomposable_score

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DECOMPOSABLE_SCORE_HPP

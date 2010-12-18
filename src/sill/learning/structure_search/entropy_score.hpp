
#ifndef SILL_LEARNING_ENTROPY_SCORE_HPP
#define SILL_LEARNING_ENTROPY_SCORE_HPP

#include <sill/learning/structure_search/decomposable_score.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Class for computing entropy for score-based structure search
   * over decomposable models.  There are 2 options:
   *  - H(X | E=e): entropy over distribution P(X | E=e) (global score)
   *  - H(X): entropy over distribution P(X) (local score)
   *
   * @param F               type of factor used in the decomposable model
   *
   * \author Joseph Bradley
   * \ingroup learning_structure
   */
  template <typename F>
  class entropy_score : public decomposable_score<F> {

  public:
    //! The base type
    typedef decomposable_score<F> base;

    using base::is_global;
    using base::has_estimate;

    //! The type of factor used in the decomposable model
    typedef F factor_type;

    //! The type of variable associated with a factor
    //    typedef typename F::variable_type variable_type;

    //! The assignment type of the factor
    typedef typename decomposable<F>::assignment_type assignment_type;

    //! The domain type of the factor
    typedef typename F::domain_type domain_type;

    /**
     * PARAMETERS
     *  - EVIDENCE (assignment_type): assignment E=e used in computing
     *     entropy H(X | E=e)
     *     (default = none, so score is the entropy H(X))
     */
    class parameters {
    private:
      assignment_type evidence_;
    public:
      parameters() { }
      parameters& evidence(assignment_type value) {
        evidence_ = value; return *this;
      }
      const assignment_type& evidence() const { return evidence_; }
    }; // class parameters

    //! Functor used by the map functions in decomposable_change.
    class entropy_score_functor : public decomposable_score_functor<F> {
    protected:
      friend class entropy_score<F>;

      //! cumulative entropy
      double H;

      entropy_score_functor() : H(0) { }

    public:
      void inserted_clique(const F& f) { H += f.entropy(); std::cerr << "inserted_clique: " << f.arguments() << std::endl; }
      void deleted_clique(const F& f) { H -= f.entropy(); std::cerr << "deleted_clique: " << f.arguments() << std::endl; }
      void inserted_separator(const F& f) { H -= f.entropy(); std::cerr << "inserted_separator: " << f.arguments() << std::endl; }
      void deleted_separator(const F& f) { H += f.entropy(); std::cerr << "deleted_separator: " << f.arguments() << std::endl; }
    }; // class entropy_score_functor

  protected:
    assignment_type evidence;

  public:
    /**
     * Constructs a decomposable score based on the entropy of P(X) or
     * the entropy of P(X | E=e) where E=e is specified by the parameters.
     */
    explicit entropy_score(parameters params = parameters())
      : base((params.evidence().size() > 0), false),
        evidence(params.evidence()) {
    }

    //! Given a model, compute the score (from scratch)
    double compute(const learnt_decomposable<F>& model) const {
      if (is_global) {
        decomposable<F> tmpmodel(model);
        tmpmodel.condition(evidence);
        return tmpmodel.entropy();
      } else {
        return model.entropy();
      }
    }

    //! Given a model, current score, and change to the model (not yet made),
    //! compute the change in the score.
    //! This takes advantage of local scores when possible.
    //! @return <true if move is valid, [new score - cur_score]>
    std::pair<bool, double>
    compute_change(const learnt_decomposable<F>& model, double cur_score,
                   const decomposable_change<F>& change,
                   statistics& stats) const {
      if (is_global) {
        if (!(change.valid(model)))
          return std::make_pair(false, 0.);
        learnt_decomposable<F> tmpmodel(model);
        change.commit(tmpmodel, stats);
        tmpmodel.condition(evidence);
        return std::make_pair(true, tmpmodel.entropy() - cur_score);
      } else {
        //The change will be:
        // entropy of added cliques and deleted separators
        // - entropy of deleted cliques and added separators
        entropy_score_functor func;
        bool valid = change.map_score_functor(func, model, stats);
        return std::make_pair(valid, func.H);
      }
    }

    //! Given a model, compute an estimate of the score.
    //! This actually just computes the exact entropy.
    double estimate(const learnt_decomposable<F>& model) const {
      return compute(model);
    }

    //! Given a model, current score, and change to the model (not yet made),
    //! estimate the change in the score.
    //! This actually just computes the exact change.
    //! @return <true if move is valid, [new score - cur_score]>
    std::pair<bool, double>
    estimate_change(const learnt_decomposable<F>& model, double cur_score,
                    const decomposable_change<F>& change,
                    statistics& stats) const {
      return compute_change(model, cur_score, change, stats);
    }

  }; // class entropy_score

} // end of namespace: prl

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_ENTROPY_SCORE_HPP

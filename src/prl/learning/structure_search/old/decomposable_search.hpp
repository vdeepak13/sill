
#ifndef PRL_LEARNING_DECOMPOSABLE_SEARCH_HPP
#define PRL_LEARNING_DECOMPOSABLE_SEARCH_HPP

#include <prl/datastructure/mutable_queue.hpp>
#include <prl/learning/structure_learning/decomposable_move.hpp>
#include <prl/model/learnt_decomposable.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  /**
   * Class for learning decomposable models via structure search
   * in which the search is constrained to a fixed set of types of moves.
   *
   * This expects an initial model.  To try lots of initial models,
   * see decomposable_iterator.
   *
   * @param F         type of factor used in the decomposable model
   * @param Inserter  Inserter.push(m, score) must insert move m with
   *                  score/estimate score into whatever container collects
   *                  new moves which are generated by this class.
   * @param MarginalSource  MarginalSource::marginal(const domain_type&) must
   *                        return a marginal over the given domain
   * @param MarginalPtr     MarginalPtr used by decomposable_change
   *
   * \author Joseph Bradley
   * \ingroup learning_structure
   */
  template <typename F, typename Inserter, typename MarginalSource,
            typename MarginalPtr = boost::shared_ptr<F> >
  class decomposable_search {

  public:
    //! The type of factor used in the decomposable model
    typedef F factor_type;

    //! The type of variable associated with a factor
    //    typedef typename F::variable_type variable_type;

    //! The domain type of the factor
    typedef typename F::domain_type domain_type;

    //! The type of edge associated with the model
    typedef typename decomposable<F>::edge edge;

    //! The type of vertex associated with the model
    typedef typename decomposable<F>::vertex vertex;

    //! The type of decomposable move
    typedef decomposable_move<F, Inserter, MarginalSource, MarginalPtr>
      decomposable_move_type;

    /**
     * PARAMETERS
     *  - INITIAL_STEPS (size_t): number of steps to take initially
     *     (default = 0)
     *  - USE_ESTIMATES (bool): If available, use estimates of scores
     *     when choosing moves.  The record of scores will still have the
     *     exact scores.  This only applies to global scores.
     *     (default = true)
     */
    class parameters {
    private:
      size_t initial_steps_;
      bool use_estimates_;
    public:
      parameters() : initial_steps_(0), use_estimates_(true) {}
      parameters& initial_steps(size_t value) {
        initial_steps_ = value; return *this;
      }
      parameters& use_estimates(bool value) {
        use_estimates_ = value; return *this;
      }
      size_t initial_steps() const { return initial_steps_; }
      size_t use_estimates() const { return use_estimates_; }
    }; // class parameters

  protected:
    parameters params;

    //! Variables to include in learned model
    domain_type variables;
    //! marginal source
    const MarginalSource& marginal_source;
    //! Score used for structure search (same as in params)
    decomposable_score score;
    //! Types of moves allowed in search
    std::vector<decomposable_move_type> allowed_moves;
    //! If true, use estimates of scores when choosng moves
    bool use_estimates;

    //! Current model
    learnt_decomposable<F> model;
    //! Current score
    double current_score_;
    //! Priority queue of available moves
    mutable_queue<decomposable_move_type,double> move_queue;
    //! Number of moves made
    size_t num_moves_;

  public:
    /**
     * Construct a structure search for decomposable models which may be
     * run all at once or one step at a time.
     *
     * @param variables        variables to include in learned model
     * @param marginal_source  dataset or statistics class
     * @param score            score used for structure search
     * @param allowed_moves    types of moves allowed in search
     * @param initial_model    initial structure
     */
    decomposable_search(const domain_type& variables,
                        const MarginalSource& marginal_source,
                        const decomposable_score& score,
                        const std::vector<decomposable_move>& allowed_moves,
                        const decomposable<F>& initial_model,
                        parameters params = parameters())
      : params(params), variables(variables), marginal_source(marginal_source),
        score(score), allowed_moves(allowed_moves),
        use_estimates(score.is_global && score.has_estimate &&
                      params.use_estimates),
        model(initial_model), current_score_(score.compute(initial_model)),
        num_moves_(0) {
      // Generate all initial moves
      for (size_t i = 0; i < allowed_moves.size(); ++i)
        allowed_moves[i].generate_all_moves(model, score, marginal_source,
                                            move_queue);
      // Run initial steps
      for (size_t i = 0; i < params.initial_steps(); ++i)
        if (!(step()))
          return;
    }

    //! Return the current model.
    const decomposable<F>& current_model() const { return model; }

    //! Return the score of the current model.
    double current_score() const { return current_score_; }

    //! Return the number of moves which have been made
    size_t num_moves() const { return num_moves_; }

    //! Choose one move and commit it.
    //! @return true iff a valid move which improved the score was found
    bool step() {
      if (move_queue.size() == 0)
        return false;
      // Find the best move.
      // The scores of many moves in the queue could be incorrect estimates,
      // but this makes sure that the chosen move as a correct estimate/score
      // which is at least as good as the other (possibly incorrect)
      // estimates/scores of moves in the queue.
      bool top_move_best = false;
      do {
        const decomposable_move_type& move = move_queue.top().first;
        if (move.valid(model)) {
          double score_change;
          if (use_estimates)
            score_change =
              move.score_change_estimate(model, score, current_score_);
          else
            score_change = move.score_change(model, score, current_score_);
          move_queue.update(move, score_change);
          if (move_queue.top().second == score_change)
            if (score_change <= 0) {
              update_all_moves();
              if (move_queue.top().second <= 0) {
                move_queue.clear();
                return false;
              } else
                top_move_best = true;
            } else
              top_move_best = true;
        } else {
          move_queue.pop();
          if (move_queue.size() == 0)
            return false;
        }
      } while (!top_move_best);
      // Commit the top move in the queue
      std::pair<decomposable_move_type,double>
        move_score_pair(move_queue.pop());
      move_score_pair.first.commit(model);
      // Generate new possible moves, score them, and add them to the queue
      for (size_t i = 0; i < allowed_moves.size(); ++i)
        allowed_moves[i].generate_new_moves(model, score,
                                            move_score_pair.first.change,
                                            marginal_source, move_queue);
      ++num_moves_;
      return true;
    }

  }; // class decomposable_search

} // end of namespace: prl

#include <prl/macros_undef.hpp>

#endif // #ifndef PRL_LEARNING_DECOMPOSABLE_SEARCH_HPP

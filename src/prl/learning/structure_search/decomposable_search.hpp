
#ifndef SILL_LEARNING_DECOMPOSABLE_SEARCH_HPP
#define SILL_LEARNING_DECOMPOSABLE_SEARCH_HPP

#include <sill/datastructure/mutable_queue.hpp>
#include <sill/learning/structure_search/decomposable_move.hpp>
#include <sill/model/learnt_decomposable.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Class for learning decomposable models via structure search
   * in which the search is constrained to a fixed set of types of moves.
   *
   * This expects an initial model.  To try lots of initial models,
   * see decomposable_iterator.
   *
   * @param F         type of factor used in the decomposable model
   *
   * \author Joseph Bradley
   * \ingroup learning_structure
   * \todo Keep track of statistics like numbers of moves generated,
   *       average score changes of new moves, etc.
   */
  template <typename F>
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

    //! The type of queue used to store moves
    typedef mutable_queue<decomposable_change<F>*, double> queue_type;

    //! The type of decomposable move
    typedef decomposable_move<F, queue_type> decomposable_move_type;

    /**
     * PARAMETERS
     *  - initial_steps (size_t): number of steps to take initially
     *     (default = 0)
     *  - use_estimates (bool): If available, use estimates of scores
     *     when choosing moves.  The record of scores will still have the
     *     exact scores.  This only applies to global scores.
     *     (default = true)
     *  - maximal_JTs (bool): Indicates if model should always have maximal-size
     *     cliques.
     *     (default = false)
     *  - max_clique_size (size_t): Maximal clique size (>0) in model.
     *     (default = 2)
     */
    class parameters {
    private:
      size_t initial_steps_;
      bool use_estimates_;
      bool maximal_JTs_;
      size_t max_clique_size_;
    public:
      parameters() : initial_steps_(0), use_estimates_(true),
                     maximal_JTs_(false), max_clique_size_(2) {}
      parameters& initial_steps(size_t value) {
        initial_steps_ = value; return *this;
      }
      parameters& use_estimates(bool value) {
        use_estimates_ = value; return *this;
      }
      parameters& maximal_JTs(bool value) {
        maximal_JTs_ = value; return *this;
      }
      parameters& max_clique_size(size_t value) {
        assert(value > 0);
        max_clique_size_ = value; return *this;
      }
      size_t initial_steps() const { return initial_steps_; }
      size_t use_estimates() const { return use_estimates_; }
      bool maximal_JTs() const { return maximal_JTs_; }
      size_t max_clique_size() const { return max_clique_size_; }
    }; // class parameters

  protected:
    parameters params;

    //! Variables to include in learned model
    domain_type variables;
    //! marginal source
    statistics& stats;
    //! Score used for structure search (same as in params)
    const decomposable_score<factor_type>& score;
    //! Types of moves allowed in search
    std::vector<decomposable_move_type*> allowed_moves;
    //! If true, use estimates of scores when choosng moves
    bool use_estimates;

    //! Current model
    learnt_decomposable<F> model;
    //! Current score
    double current_score_;
    //! Priority queue of available moves
    queue_type move_queue;
    //! Number of moves made
    size_t num_moves_;

    //! Update the scores of all moves in the queue.
    //! This actually just re-generates all moves since it's probably more
    //! efficient.
    void update_all_moves() {
      move_queue.clear();
      for (size_t i = 0; i < allowed_moves.size(); ++i)
        allowed_moves[i]->generate_all_moves(model, current_score_, score,
                                             stats, move_queue,
                                             params.use_estimates());
    }

    //! Clear the move queue
    void clear_move_queue() {
      while (move_queue.size() > 0) {
        std::pair<decomposable_change<F>*,double>
          move_score_pair(move_queue.pop());
        delete(move_score_pair.first);
      }
    }

  public:
    /**
     * Construct a structure search for decomposable models which may be
     * run all at once or one step at a time.
     *
     * @param variables        variables to include in learned model
     * @param stats            statistics class for computing marginals
     * @param score            score used for structure search
     * @param allowed_moves    types of moves allowed in search
     * @param initial_model    initial structure
     */
    decomposable_search(const domain_type& variables, statistics& stats,
                        const decomposable_score<factor_type>& score,
                        const std::vector<decomposable_move_type*>&
                        allowed_moves,
                        const decomposable<F>& initial_model,
                        parameters params = parameters())
      : params(params), variables(variables), stats(stats),
        score(score), allowed_moves(allowed_moves),
        use_estimates(score.is_global && score.has_estimate &&
                      params.use_estimates()),
        num_moves_(0) {
      // Initialize model and current score
      typename learnt_decomposable<F>::parameters ld_params;
      ld_params.maximal_JTs(params.maximal_JTs())
        .max_clique_size(params.max_clique_size());
      model = learnt_decomposable<F>(initial_model, ld_params);
      current_score_ = score.compute(model);
      // Generate all initial moves
      for (size_t i = 0; i < allowed_moves.size(); ++i)
        allowed_moves[i]->generate_all_moves(model, current_score_, score,
                                             stats, move_queue,
                                             params.use_estimates());
      // Run initial steps
      for (size_t i = 0; i < params.initial_steps(); ++i)
        if (!(step()))
          return;
    }

    ~decomposable_search() {
      clear_move_queue();
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
        decomposable_change<F>* move_ptr = move_queue.top().first;
        const decomposable_change<F>& move = *move_ptr;
        bool valid;
        double score_change;

        if (use_estimates) {
          boost::tie(valid, score_change) =
            score.estimate_change(model, current_score_, move, stats);
        } else {
          boost::tie(valid, score_change) =
            score.compute_change(model, current_score_, move, stats);
        }

        if (valid) {
          move_queue.update(move_ptr, score_change);
          if (move_queue.top().second == score_change) {
            if (score_change <= 0) {
              update_all_moves();
              if (move_queue.top().second <= 0) {
                clear_move_queue();
                return false;
              } else {
                top_move_best = true;
              }
            } else {
              top_move_best = true;
            }
          }
        } else {
          delete(move_ptr);
          move_queue.pop();
          if (move_queue.size() == 0)
            return false;
        }
      } while (!top_move_best);
      // Commit the top move in the queue.
      // Generate new possible moves, score them, and add them to the queue.
      std::pair<decomposable_change<F>*,double>
        move_score_pair(move_queue.pop());
      std::vector<typename decomposable_change<F>::clique_change> clique_changes
        = move_score_pair.first->commit(model, stats);
      current_score_ += move_score_pair.second;
      for (size_t i = 0; i < allowed_moves.size(); ++i)
        allowed_moves[i]->generate_new_moves(model, current_score_, score,
                                             clique_changes, stats, move_queue);
      ++num_moves_;
      return true;
    }

  }; // class decomposable_search

} // end of namespace: prl

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DECOMPOSABLE_SEARCH_HPP

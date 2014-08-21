#ifndef SILL_LEARNING_DMOVE_SWAP_VARIABLES_HPP
#define SILL_LEARNING_DMOVE_SWAP_VARIABLES_HPP

#include <set>
#include <sill/learning/structure_search/decomposable_move.hpp>
#include <sill/base/stl_util.hpp>
#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Class for representing a possible move in structure search for
   * decomposable models:
   *  Swap two variables in the model.
   *
   * For maximal and non-maximal JTs.
   *
   * @param F         type of factor used in the decomposable model
   * @param Inserter  Inserter.push(m, score) must insert move m with
   *                  score/estimate score into whatever container collects
   *                  new moves which are generated by this class.
   *
   * \author Joseph Bradley
   * \ingroup learning_structure
   */
  template <typename F, typename Inserter>
  class dmove_swap_variables
    : public decomposable_move<F,Inserter> {

  public:
    //! The base (move) type
    typedef decomposable_move<F,Inserter> base;

    //! The type of factor used in the decomposable model
    typedef F factor_type;

    //! The type of variable associated with a factor
    typedef typename F::variable_type variable_type;

    //! The domain type of the factor
    typedef typename base::domain_type domain_type;

    //! The generic representation for clique changes
    typedef typename base::clique_change clique_change;

    //! The type of edge associated with the model
    typedef typename base::edge edge;

    //! The type of vertex associated with the model
    typedef typename base::vertex vertex;

    //////////////////////// PRIVATE DATA AND METHODS ///////////////////////

  private:
    //! Variables to swap
    variable_type* var1;
    variable_type* var2;

    ////////////////////////// PUBLIC METHODS //////////////////////////////
  public:
    //! Creates a null move (which can be used to generate new moves).
    dmove_swap_variables() : var1(NULL), var2(NULL) { }

    //! Creates a move which swaps 2 variables.
    dmove_swap_variables(variable_type* var1, variable_type* var2)
      : var1(var1), var2(var2) {
    }

    //! Given a model, score, and statistics class, generate all possible moves
    //! and insert pointers to them (and their scores) into the provided
    //! Inserter.
    void
    generate_all_moves(const learnt_decomposable<F>& model, double cur_score,
                       const decomposable_score<F>& score, dataset_statistics<>& stats,
                       Inserter& inserter, bool use_estimates = false) const {
      // For each pair of variables
      std::vector<variable_type*> var_vec;
      foreach(variable_type* v, model.arguments())
        var_vec.push_back(v);
      for (size_t i = 0; i < var_vec.size() - 1; ++i) {
        for (size_t j = i+1; j < var_vec.size(); ++j) {
          dmove_swap_variables<F,Inserter>* move_ptr =
            new dmove_swap_variables<F,Inserter>(var_vec[i], var_vec[j]);
          double change;
          if (use_estimates)
            change =
              score.estimate_change(model, cur_score, *move_ptr, stats).second;
          else
            change
              = score.compute_change(model, cur_score, *move_ptr, stats).second;
          inserter.push(move_ptr, change);
        }
      }
    }

    //! Given a model, score, another decomposable_move (which has just been
    //! committed), and statistics class,
    //! generate all possible new moves of this type and insert pointers to
    //! them (and their scores) into the provided Inserter.
    void
    generate_new_moves(const learnt_decomposable<F>& model, double cur_score,
                       const decomposable_score<F>& score,
                       const std::vector<clique_change>& clique_changes,
                       dataset_statistics<>& stats, Inserter& inserter,
                       bool use_estimates = false) const {
      // TODO:
      // This should really only update moves since all possible moves will
      //  already be in the queue (unless one of these types of moves has just
      //  been committed).  The best way to do this would be to keep separate
      //  queues for each type of move, but let's save that for later.

      // If we do this the current way, then any of these options would work:
      //  1) For each variable which has been added to a clique, generate all
      //     moves involving that variable. (minimum)
      //  2) Generate moves for each variable which has been added/deleted.
      //  3) Generate moves for all variables which are in changed cliques.
      //     (update all scores)
      // Let's do the last for now.
      domain_type affected_vars;
      foreach(const clique_change& change, clique_changes) {
        affected_vars = set_union(affected_vars, model.clique(change.v));
      }
      std::vector<variable_type*> var_vec(affected_vars.begin(),
                                          affected_vars.end());
      for (size_t i = 0; i < var_vec.size() - 1; ++i) {
        for (size_t j = i+1; j < var_vec.size(); ++j) {
          dmove_swap_variables<F,Inserter>* move_ptr =
            new dmove_swap_variables<F,Inserter>(var_vec[i], var_vec[j]);
          double change;
          if (use_estimates)
            change =
              score.estimate_change(model, cur_score, *move_ptr, stats).second;
          else
            change
              = score.compute_change(model, cur_score, *move_ptr, stats).second;
          inserter.push(move_ptr, change);
        }
      }
      foreach(variable_type* v1, var_vec) {
        foreach(variable_type* v2, set_difference(model.arguments(), affected_vars)) {
          dmove_swap_variables<F,Inserter>* move_ptr =
            new dmove_swap_variables<F,Inserter>(v1, v2);
          double change;
          if (use_estimates)
            change =
              score.estimate_change(model, cur_score, *move_ptr, stats).second;
          else
            change
              = score.compute_change(model, cur_score, *move_ptr, stats).second;
          inserter.push(move_ptr, change);
        }
      }
    }

    //! Call a score functor on each clique/separator inserted/deleted
    //! by this move (where the move is represented by clique/separator
    //! insertions/deletions).
    //! @return false if move is invalid, else true
    bool map_score_functor(decomposable_score_functor<F>& func,
                           const learnt_decomposable<F>& model,
                           dataset_statistics<>& stats) const {
      if (var1 == NULL || var2 == NULL)
        return false;
      foreach(vertex v, model.vertices()) {
        domain_type c(model.clique(v));
        if (c.count(var1) && !(c.count(var2))) {
          c.erase(var1);
          c.insert(var2);
        } else if (c.count(var2) && !(c.count(var1))) {
          c.erase(var2);
          c.insert(var1);
        } else
          continue;
        func.deleted_clique(model.marginal(v));
        func.inserted_clique(stats.marginal(c));
      }
      foreach(edge e, model.edges()) {
        domain_type s(model.separator(e));
        if (s.count(var1) && !(s.count(var2))) {
          s.erase(var1);
          s.insert(var2);
        } else if (s.count(var2) && !(s.count(var1))) {
          s.erase(var2);
          s.insert(var1);
        } else
          continue;
        func.deleted_separator(model.marginal(e));
        func.inserted_separator(stats.marginal(s));
      }
      return true;
    }

    //! Given a model, check to see if the move is still valid.
    bool valid(const learnt_decomposable<F>& model) const {
      if (var1 == NULL || var2 == NULL)
        return false;
      return true;
    }

    //! Given a mutable model, commit the move.
    //! Note: This does NOT check if the move is valid!
    //! Also, this does not calibrate or renormalize the model.
    std::vector<clique_change>
    commit(learnt_decomposable<F>& model, dataset_statistics<>& stats) const {
      std::vector<clique_change> changes;
      foreach(vertex v, model.vertices()) {
        domain_type c(model.clique(v));
        if (c.count(var1) && !(c.count(var2))) {
          c.erase(var1);
          c.insert(var2);
          domain_type var1_domain; var1_domain.insert(var1);
          domain_type var2_domain; var2_domain.insert(var2);
          changes.push_back(clique_change(v, var1_domain, var2_domain));
        } else if (c.count(var2) && !(c.count(var1))) {
          c.erase(var2);
          c.insert(var1);
          domain_type var1_domain; var1_domain.insert(var1);
          domain_type var2_domain; var2_domain.insert(var2);
          changes.push_back(clique_change(v, var2_domain, var1_domain));
        } else
          continue;
        model.set_clique(v, c, stats.marginal(c));
      }
      return changes;
    }

  }; // class dmove_swap_variables

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DMOVE_SWAP_VARIABLES_HPP

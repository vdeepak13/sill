
#ifndef PRL_LEARNING_DECOMPOSABLE_MOVE_HPP
#define PRL_LEARNING_DECOMPOSABLE_MOVE_HPP

#include <prl/learning/dataset/dataset.hpp>
#include <prl/learning/structure_learning/decomposable_score.hpp>
#include <prl/model/learnt_decomposable.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  /**
   * Class for representing a possible move in structure search for
   * decomposable models.
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
  class decomposable_move {

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

    decomposable_move() {
      assert(false);
    }

    virtual ~decomposable_move() { }

    //! Change to the model in this move
    const decomposable_change change;

    //! Given a model, score, and MarginalSource, generate all possible moves
    //! and insert them (and their scores) into the provided Inserter.
    virtual static void
    generate_all_moves(const decomposable<F>& model,
                       const decomposable_score<F,MarginalPtr>& score,
                       const MarginalSource& marginal_source,
                       Inserter& inserter) = 0;

    //! Given a model, score, decomposable_change (which has just been
    //! committed), and MarginalSource,
    //! generate all possible new moves of this type and insert them
    //! (and their scores) into the provided Inserter.
    virtual static void
    generate_new_moves(const decomposable<F>& model,
                       const decomposable_score<F,MarginalPtr>& score,
                       const decomposable_change& move,
                       const MarginalSource& marginal_source,
                       Inserter& inserter) = 0;

    //! Return the score change via an estimate.
    //! If the score is local, then this computes the exact score change.
    //! If the score is global, then this computes an estimate if possible
    //! and an exact score change otherwise.
    //! @return [new score - current score]
    double score_change_estimate(const decomposable<F>& model,
                                 const decomposable_score<F,MarginalPtr>& score,
                                 double cur_score) const {
      if (score.is_global) {
        if (score.has_estimate)
          return score.estimate_change(model, cur_score, change);
        else
          return score.compute(model) - cur_score;
      } else
        return score.compute_change(model, cur_score, change);
    }

    //! Return the exact score change.
    //! This takes advantage of local scores when possible.
    //! @return [new score - old score]
    double score_change(const decomposable<F>& model,
                        const decomposable_score<F,MarginalPtr>& score,
                        double cur_score) const {
      if (score.is_global)
        return score.compute(model) - cur_score;
      else
        return score.compute_change(model, cur_score, change);
    }

    //! Given a model, check to see if the move is still valid.
    bool valid(const decomposable<F>& model) const {
      for (size_t i = 0; i < change.vertex_checks.size(); ++i) {
        const vertex_check& check = change.vertex_checks[i];
        if (check.required_variables_exact) {
          if (model.clique(check.vertex_ID) != check.required_variables)
            return false;
        } else {
          if (check.required_variables.minus(model.clique(check.vertex_ID)).
              size() > 0)
            return false;
        }
        domain_type required(check.required_neighbor_variables);
        domain_type disallowed(check.disallowed_neighbor_variables);
        foreach(const vertex& u, model.neighbors(check.vertex_ID)) {
          const domain_type& u_vars = model.clique(u);
          required.remove(u_vars);
          if (disallowed.meets(u_vars))
            return false;
        }
        if (required.size() > 0)
          return false;
      }
      return true;
    }

    //! Given a mutable model, commit the move.
    //! Note: This does NOT check if the move is valid!
    //! Also, this does not calibrate or renormalize the model.
    void commit(learnt_decomposable<F>& model) const {
      assert(false);

      std::set<edge> edges_to_be_updated;
      std::vector<vertex> vertex_order;

      for (size_t i = 0; i < change.vertex_changes.size(); ++i) {
        const vertex_change& vchange = change.vertex_changes[i];
        if (vchange.new_vertex) {
          // Create a new clique with insert_variables in model.jt
          // Set the potential for the new vertex using marginal_ptr
          vertex new_v(model.add_clique(vchange.insert_variables,
                                        *(vchange.marginal_ptr)));
          vertex_order.push_back(new_v);
        } else {
          vertex u(vchange.vertex_ID);
          vertex_order.push_back(u);
          assert(model.contains(u));
          // Set clique variables to be
          //  model.clique(u).minus(vchange.delete_variables)
          //   .plus(vchange.insert_variables)
          // Set clique potential to be
          //  vchange.marginal_ptr.operator*()
          //   .collapse(new clique variables, sum_op)
          domain_type new_domain(model.clique(u).minus(vchange.delete_variables)
                                 .plus(vchange.insert_variables));
          model.set_clique(u, new_domain,
                           vchange.marginal_ptr.operator*().collapse(new_domain,
                                                                     sum_op));
          // Add edges to edges_to_be_updated
          foreach(const edge& e, model.out_edges(u))
            edges_to_be_updated.insert(e);
        }
      }
      // Insert new edges (and set their potentials)
      for (size_t i = 0; i < change.insert_edges1.size(); ++i)
        model.add_edge(change.insert_edges1[i]);
      for (size_t i = 0; i < change.insert_edges2.size(); ++i)
        model.add_edge(change.insert_edges2[i].first,
                       vertex_order[change.insert_edges2[i].second]);
      for (size_t i = 0; i < change.insert_edges3.size(); ++i)
        model.add_edge(vertex_order[change.insert_edges3[i].first],
                       vertex_order[change.insert_edges3[i].second]);
      // Update/delete other edges as necessary
      foreach(const edge& e, edges_to_be_updated)
        model.update_edge(e);
    }

  }; // class decomposable_move

} // end of namespace: prl

#include <prl/macros_undef.hpp>

#endif // #ifndef PRL_LEARNING_DECOMPOSABLE_MOVE_HPP

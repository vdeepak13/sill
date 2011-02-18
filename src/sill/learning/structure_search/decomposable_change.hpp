
#ifndef SILL_LEARNING_DECOMPOSABLE_CHANGE_HPP
#define SILL_LEARNING_DECOMPOSABLE_CHANGE_HPP

#include <sill/learning/dataset/dataset_statistics.hpp>
#include <sill/learning/structure_search/decomposable_score_functor.hpp>
#include <sill/model/learnt_decomposable.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Virtual class for representing a change in a decomposable model.
   * This is used to simplify dependencies among decomposable_move and
   * decomposable_score.
   *
   * @param F         type of factor used in the decomposable model
   *
   * \author Joseph Bradley
   * \ingroup learning_structure
   */
  template <typename F>
  class decomposable_change {

  public:
    //! The type of factor used in the decomposable model
    typedef F factor_type;

    //! The type of variable associated with a factor
    typedef typename F::variable_type variable_type;

    //! The domain type of the factor
    typedef typename F::domain_type domain_type;

    //! The type of vertex associated with the model
    typedef typename decomposable<F>::vertex vertex;

    //! Class for storing changes (already made) to a model
    class clique_change {
    public:
      clique_change(vertex v, domain_type deleted_vars, domain_type added_vars)
        : v(v), deleted_vars(deleted_vars), added_vars(added_vars) { }
      vertex v;
      domain_type deleted_vars;
      domain_type added_vars;
    }; // class clique_change

    virtual ~decomposable_change() { }

    //! Call a score functor on each clique/separator inserted/deleted
    //! by this move (where the move is represented by clique/separator
    //! insertions/deletions).
    //! @return false if move is invalid
    //! @todo This could be made to be more efficient by using this to check
    //!       for validity as well.
    virtual bool map_score_functor(decomposable_score_functor<F>& func,
                                   const learnt_decomposable<F>& model,
                                   dataset_statistics& stats) const = 0;

    //! Given a model, check to see if the move is still valid.
    virtual bool valid(const learnt_decomposable<F>& model) const = 0;

    //! Given a mutable model, commit the move.
    //! Note: This does NOT check if the move is valid!
    //! Also, this does not calibrate or renormalize the model.
    //! @return list of clique changes, providing a generic description of
    //!         the changes to the model
    virtual std::vector<clique_change>
    commit(learnt_decomposable<F>& model, dataset_statistics& stats) const = 0;

  }; // class decomposable_change

} // end of namespace: prl

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DECOMPOSABLE_CHANGE_HPP

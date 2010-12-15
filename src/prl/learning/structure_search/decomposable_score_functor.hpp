
#ifndef PRL_LEARNING_DECOMPOSABLE_SCORE_FUNCTOR_HPP
#define PRL_LEARNING_DECOMPOSABLE_SCORE_FUNCTOR_HPP

#include <prl/macros_def.hpp>

namespace prl {

  /**
   * Virtual functor used by the map functions in decomposable_change.
   * Actual score classes should create subclasses of this class.
   * @param F     type of factor used in the decomposable model
   */
  template <typename F>
  class decomposable_score_functor {
  public:
    virtual ~decomposable_score_functor() { }
    //! Do computation on the marginal for an inserted clique.
    virtual void inserted_clique(const F& f) = 0;
    //! Do computation on the marginal for a deleted clique.
    virtual void deleted_clique(const F& f) = 0;
    //! Do computation on the marginal for an inserted separator.
    virtual void inserted_separator(const F& f) = 0;
    //! Do computation on the margvinal for a deleted separator.
    virtual void deleted_separator(const F& f) = 0;
  }; // class decomposable_score_functor

} // end of namespace: prl

#include <prl/macros_undef.hpp>

#endif // #ifndef PRL_LEARNING_DECOMPOSABLE_SCORE_FUNCTOR_HPP

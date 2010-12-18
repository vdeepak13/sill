#ifndef SILL_GRAPHICAL_MODEL_HPP
#define SILL_GRAPHICAL_MODEL_HPP
#include <map>

#include <sill/global.hpp>
#include <sill/math/logarithmic.hpp>
#include <sill/factor/concepts.hpp>
#include <sill/model/markov_graph.hpp>

#include <sill/range/forward_range.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! \addtogroup model
  //! @{

  /**
   * The base class of all factorized models.
   * @tparam F the type of factors stored in this model;
   *           must implement the Factor concept.
   */
  template <typename F>
  class factorized_model {
    concept_assert((Factor<F>));

  public:
    //! The underlying factor type
    typedef F factor_type;

    //! The type of variables that form the factor's domain
    typedef typename F::variable_type variable_type;

    //! The underlying domain type of a factor
    typedef typename F::domain_type domain_type;

    //! The assignment type of the factor
    typedef std::map<variable_type*, typename variable_type::value_type>
      assignment_type;

    //! The record type of the factor
    typedef typename F::record_type record_type;

    //! Returns the arguments of the graphical model
    virtual domain_type arguments() const = 0;

    /**
     * Returns the collection of factors in this graphical model.
     * The reference may be to a temporary inside the iterator range.
     */
    virtual forward_range<const F&> factors() const = 0;

    //! Returns the log-likelihood of an assignment
    //! If the model is not normalized, this returns the unnormalized log
    //! likelihood.
    virtual double log_likelihood(const assignment_type& a) const = 0;

    //! Returns the likelihood of an assignment in the logarithmic domain.
    //! If the model is not normalized, this returns the unnormalized log
    //! likelihood.
    virtual logarithmic<double> operator()(const assignment_type& a) const = 0;
    virtual ~factorized_model() {}
  }; // class factorized_model

  /**
   * The base class of all graphical models.
   * @tparam F the type of factors stored in this model;
   *           must implement the Factor concept.
   */
  template <typename F>
  class graphical_model : public factorized_model<F> {
    concept_assert((Factor<F>));
  public:
    typedef factorized_model<F> base;
    typedef typename base::factor_type       factor_type;
    typedef typename base::variable_type     variable_type;
    typedef typename base::domain_type       domain_type;
    typedef typename base::assignment_type   assignment_type;
    typedef typename base::record_type       record_type;

    //! Returns a minimal markov graph that captures the dependencies
    //! in this model
    virtual sill::markov_graph<variable_type*> markov_graph() const = 0;

    //! Determines whether x and y are separated, given z
    virtual bool
    d_separated(const domain_type& x, const domain_type& y,
                const domain_type& z = domain_type::empty_set) const = 0;
  };

  //! @} group model

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

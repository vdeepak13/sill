#ifndef SILL_GRADIENT_OBJECTIVE_HPP
#define SILL_GRADIENT_OBJECTIVE_HPP

namespace sill {

  /**
   * An interface that represents an objective that can compute its gradient.
   */
  template <typename Vec>
  class gradient_objective {
  public:
    //! The storage type of the vector
    typedef typename Vec::value_type real_type;
    
    /**
     * Default constructor.
     */
    gradient_objective() { }

    /**
     * Destructor.
     */
    virtual ~gradient_objective() { }

    /**
     * Computes the value of the objective for the given input.
     */
    virtual real_type value(const Vec& x) = 0;

    /**
     * Computes the gradient of the objective for the given input.
     */
    virtual const Vec& gradient(const Vec& x) = 0;

    /**
     * Computes the preconditioned gradient of the objective.
     */
    virtual const Vec& hessian_diag(const Vec& x) = 0;

  }; // class gradient_objective

} // namespace sill

#endif

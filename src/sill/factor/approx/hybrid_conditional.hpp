#ifndef SILL_HYBRID_CONDITIONAL_APPROXIMATOR_HPP
#define SILL_HYBRID_CONDITIONAL_APPROXIMATOR_HPP

#include <boost/shared_ptr.hpp>

#include <sill/math/constants.hpp>
#include <sill/factor/approx/interfaces.hpp>
#include <sill/factor/moment_gaussian.hpp>

namespace sill {

  /**
   * A Hybrid conditional approximator.
   * The approximator splits the domain for one of the variables,
   * and applies a standard linearization for the remaining variables.
   * \ingroup factor_approx
   */
  class hybrid_conditional_approximator : public gaussian_approximator {
    
  private:
    //! The base approximator
    boost::shared_ptr<gaussian_approximator> approx_ptr;
    
    //! The variable being split
    vector_variable* split_var;

    //! The number of points
    size_t npoints;

    //! The minimum standard deviation required to perform a split
    double minstdev;

    //! The number of standard deviations to integrate over
    double nstdevs;

    //! The maximum range to integrate over
    double max_range;

    //! Returns a reference to the base approximator
    const gaussian_approximator& approx() const {
      return *approx_ptr;
    }

  public:
    /**
     * Constructs a hybrid conditional approximator.
     * \param base The base approximator
     * \param split_var The variable being split on
     * \param npoints The number of split points
     * \param minstdev The minimum number of std. deviations to perform a split
     * \param nstdevs The number of standard deviations to cover
     */
    hybrid_conditional_approximator(const gaussian_approximator& base,
                                    vector_variable* split_var, 
                                    size_t npoints,
                                    double minstdev = 0,
                                    double nstdevs = 2,
                                    double max_range = inf<double>());
    
    hybrid_conditional_approximator* clone() const {
      return new hybrid_conditional_approximator(*this);
    }

    moment_gaussian operator()(const nonlinear_gaussian& ng,
                               const moment_gaussian& prior) const;

  }; // class hybrid_conditional_approximator
  
} // namespace sill

#endif

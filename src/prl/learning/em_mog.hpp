#ifndef SILL_EM_MOG_HPP
#define SILL_EM_MOG_HPP

#include <sill/factor/mixture.hpp>
#include <sill/factor/moment_gaussian.hpp>
#include <sill/learning/dataset/dataset.hpp>
#include <sill/learning/parameter/gaussian.hpp>

/*
Todo: Fix mixture. Fix moment gaussians. Learning of moment gaussians. 
Test the momemnt gaussians more.
Think through weighted combinations.
*/

namespace sill {

  /**
   * implements EM with mixture of Gaussians
   * \todo Deal with normalization issues
   * \ingroup learning_param
   */
  class em_mog {

    //! The dataset
    const dataset* data;

    //! The number of points in the dataset;
    size_t n;

    //! The number of components.
    size_t k;
    
    //! The components for the current iteration
    mat w;

  public:
    //! Initializes the engine.
    em_mog(const dataset* data, size_t k);
    
    //! Computes the initial mixtures
    //! \param regul regularization parameter
    template <typename Engine>
    mixture_gaussian initialize(Engine& engine, double regul = 0) const {
      // Compute the second-order statistics of the dataset
      moment_gaussian mg;
      mle(*data, regul, mg);

      // Choose random datapoints and assign the center to them
      mixture_gaussian mixture(k, mg);
      for(size_t i = 0; i < k; i++) {
        mixture[i].mean() = data->sample(engine).vector();
      }
      
      return mixture.normalize();
    }
    
    //! Computes the probability of each datapoint for each mixture component
    //! \return the log-likelihood of the dataset
    double expectation(const mixture_gaussian& mixture);

    //! Computes the new mixture using the current components
    //! each mixture component is a weighted estimate of the datapoints.
    mixture_gaussian maximization(double regul = 0) const;

    //! Computes the new mixture but does not renormalize the mixture nor
    //! the component parameters
    mixture_gaussian local_maximization() const;

  }; // class em_mog

} // namespace sill

#endif 

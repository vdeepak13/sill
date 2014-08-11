#ifndef SILL_MIXTURE_EM3_HPP
#define SILL_MIXTURE_EM3_HPP

#include <sill/factor/mixture.hpp>
#include <sill/factor/moment_gaussian.hpp>
#include <sill/learning/mle/mle.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // the dataset size must not change after initialization
  template <typename F>
  class mixture_em {
  public:
    typedef typename F::domain_type            domain_type;
    typedef double                             storage_type; // TODO
    typedef typename mle<F>::dataset_type      dataset_type;
    typedef typename mle<F>::param_type        param_type;
    typedef typename dataset_type::record_type record_type;
    
    typedef arma::Mat<storage_type> mat_type;
    typedef arma::Col<storage_type> vec_type;

    //! Creates the engine, initializing it to the given mixture
    mixture_em(const dataset_type* dataset,
               const mixture<F>& init,
               const param_type& params = param_type())
      : dataset(dataset),
        model(init),
        estim(dataset, params),
        weights(dataset->size(), model.size()) { }
      
    //! Expectation step: computes the likelihood of data for each components.
    //! \return the log-likelihood of the current model
    storage_type expectation() {
      for (size_t c = 0; c < model.size(); ++c) {
        const F& factor = model[c];
        storage_type* w = weights.colptr(c);
        foreach(const record_type& r, dataset->records(factor.head())) { // TODO
          *w++ = factor(r.values); // TODO: weights
        }
      }
      
      vec_type norms = sum(weights, 1);
      for (size_t c = 0; c < model.size(); ++c) {
        weights.col(c) /= norms;
      }
      return sum(log(norms));
    }
    
    //! Maximization step: recomputes the components.
    void maximization() {
      for (size_t c = 0; c < model.size(); ++c) {
        model[c] = estim(model.arguments(), weights.col(c));
      }

      // the standard (non-logarithmic) implementation of normalize works
      // since our probabilities are O(data->size())
      model.normalize();
    }

    //! Returns the current estimate
    const mixture<F>& estimate() const {
      return model;
    }

  private:
    const dataset_type* dataset;
    mixture<F>  model;
    mle<F>      estim;
    mat_type    weights; // the weight matrix nsamples x num_components
    
  }; // class mixture_em

  // random initialization for mixtures of Gaussian
  inline void initialize_em(const vector_dataset<>& ds,
                            size_t num_components,
                            const vector_domain& args,
                            mixture_gaussian& mog) {
    assert(num_components > 0);
    assert(ds.size() >= num_components);

    // generate the row indices that will determine the centers
    // we want to be absolutely certain they are distinct.
    // otherwise, EM will not compute distinct clusters
    boost::mt19937 rng;
    boost::uniform_int<size_t> uniform(0, num_components - 1);
    boost::unordered_set<size_t> random_rows;
    while (random_rows.size() < num_components) {
      random_rows.insert(uniform(rng));
    }

    // compute the covariance and set the random means
    mle<moment_gaussian> estim(&ds);
    mog = mixture_gaussian(num_components, estim(args));
    size_t k = 0;
    foreach(size_t row, random_rows) {
      mog[k++].mean() = ds.record(row, mog[0].head()).values;
    }
  }

  inline void initialize_em(const vector_dataset<>& ds,
                            size_t num_components,
                            mixture_gaussian& mog) {
    initialize_em(ds, num_components, ds.arguments(), mog);
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

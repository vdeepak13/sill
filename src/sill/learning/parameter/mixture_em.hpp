#ifndef SILL_MIXTURE_EM_HPP
#define SILL_MIXTURE_EM_HPP

#include <sill/factor/mixture.hpp>
#include <sill/factor/moment_gaussian.hpp>
#include <sill/learning/parameter/factor_estimator.hpp>

#include <boost/function.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // the dataset size must not change after initialization
  template <typename F, typename Dataset>
  class mixture_em {
  public:
    typedef typename F::domain_type       domain_type;
    typedef double                        storage_type; // TODO fix this
    typedef typename Dataset::record_type record_type;
    
    typedef arma::Mat<storage_type> mat_type;
    typedef arma::Col<storage_type> vec_type;

    typedef boost::function<F(const domain_type&, const vec_type&)> learner_fn;
    typedef boost::function<F(const domain_type&)> generator_fn;

    //! Creates the engine, initializing it to the given mixture
    mixture_em(const Dataset* dataset,
               const learner_fn& flearn,
               const mixture<F>& init)
      : dataset(dataset),
        flearn(flearn),
        model(init),
        weights(dataset->size(), model.size()) { }

    //! Creates the engine, initializing it with the given component generator
    mixture_em(const Dataset* dataset,
               const learner_fn& flearn,
               size_t num_components,
               const domain_type& args,
               const generator_fn& fgen)
      : dataset(dataset),
        flearn(flearn),
        model(num_components, args),
        weights(dataset->size(), model.size()) {
      for (size_t c = 0; c < model.size(); ++c) {
        model[c] = fgen(args);
      }
      model.normalize();
    }

    //! Expectation step: computes the likelihood of data for each components.
    //! \return the log-likelihood of the current model
    storage_type expectation() {
      for (size_t c = 0; c < model.size(); ++c) {
        const F& factor = model[c];
        storage_type* w = weights.colptr(c);
        foreach(const record_type& r, dataset->records(factor.head())) { //TODO
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
        model[c] = flearn(model.arguments(), weights.col(c));
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
    mixture<F> model;
    const Dataset* dataset;
    learner_fn flearn;
    mat_type weights; // the weight matrix nsamples x num_components

  }; // class mixture_em

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

#ifndef SILL_MIXTURE_EM_HPP
#define SILL_MIXTURE_EM_HPP

#include <sill/factor/factor_evaluator.hpp>
#include <sill/factor/factor_mle_incremental.hpp>
#include <sill/factor/mixture.hpp>
#include <sill/learning/parameter/mixture_init.hpp>

#include <iostream>
#include <numeric>

#include <sill/macros_def.hpp>

namespace sill {

  // the dataset size must not change after initialization
  template <typename F>
  class mixture_em {
  public:
    // Member types of the Learner concept
    typedef typename F::real_type                          real_type;
    typedef mixture<F>                                     model_type;
    typedef typename F::dataset_type                       dataset_type;
    typedef typename factor_mle_incremental<F>::param_type comp_param_type;

    struct param_type {
      comp_param_type comp_params;
      size_t max_iters;
      real_type tolerance;
      unsigned seed;  // for initialization
      bool verbose;

      // allowing implicit conversion from comp_param_type
      param_type(const comp_param_type& params = comp_param_type(),
                 size_t max_iters = 100,
                 real_type tolerance = 1e-6,
                 unsigned seed = 0,
                 bool verbose = false)
        : comp_params(params),
          max_iters(max_iters),
          tolerance(tolerance),
          seed(seed),
          verbose(verbose) { }
    };

    // Other types
    typedef typename F::var_vector_type var_vector_type;
    typedef typename dataset_type::record_type record_type;

  public:
    //! Creates the learner for the given component count and arguments
    mixture_em(size_t num_components, const var_vector_type& args)
      : num_components(num_components), args(args) { }

    //! Learns a mixture with default parameters
    real_type learn(const dataset_type& ds, mixture<F>& model) {
      return learn(ds, param_type(), model);
    }
    
    //! Learns a mixture from the given dataset and parameters
    real_type learn(const dataset_type& ds,
                    const param_type& params,
                    mixture<F>& model) {
      initialize(&ds, params.comp_params, params.seed);
      real_type objective = std::numeric_limits<real_type>::infinity();
      for (size_t it = 0; it < params.max_iters; ++it) {
        real_type old = objective;
        objective = iterate();
        if (params.verbose) {
          std::cout << "Iteration " << it << ": ll=" << objective << std::endl;
        }
        if (it > 1 && std::abs(objective - old) / ds.size() < params.tolerance) {
          break;
        }
      }
      model = this->model;
      return objective;
    }

    //! Initializes the internal state
    void initialize(const dataset_type* dataset,
                    const comp_param_type& params = comp_param_type(),
                    unsigned seed = 0) {
      this->dataset = dataset;
      this->params = params;
      sill::initialize(num_components, args, *dataset, params, seed, model);
    }

    //! Performs one expectation and one maximization step
    //! \return (the lower bound) of the log-likelihood of the previous model
    real_type iterate() {
      // initialize the evaluators and factor estimators for each component
      std::vector<factor_evaluator<F> >       evaluators;
      std::vector<factor_mle_incremental<F> > estimators;
      evaluators.reserve(num_components);
      estimators.reserve(num_components);
      for (size_t i = 0; i < num_components; ++i) {
        evaluators.push_back(factor_evaluator<F>(model[i]));
        estimators.push_back(factor_mle_incremental<F>(args, params));
      }

      std::vector<real_type> p(num_components);
      real_type bound = 0.0;
      foreach (const record_type& r, dataset->records(args)) {
        // compute the probability of the datapoint under each component
        for (size_t i = 0; i < num_components; ++i) {
          p[i] = evaluators[i](r.values);
        }
        real_type sump = std::accumulate(p.begin(), p.end(), real_type());
        real_type mult = r.weight / sump;
        bound += r.weight * std::log(sump);
        
        // update the estimates
        for (size_t i = 0; i < num_components; ++i) {
          estimators[i].process(r.values, p[i] * mult);
        }
      }
      
      // recompute the components
      for (size_t i = 0; i < num_components; ++i) {
        model[i] = estimators[i].estimate();
        model[i] *= estimators[i].weight();
      }
      model.normalize();

      return bound;
    }

    //! Returns the current estimate
    const mixture<F>& estimate() const {
      return model;
    }

  private:
    // persistent members
    size_t num_components;
    var_vector_type args;
    
    // iteration-specific members
    const dataset_type* dataset;
    comp_param_type params;
    mixture<F> model;
    
  }; // class mixture_em

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

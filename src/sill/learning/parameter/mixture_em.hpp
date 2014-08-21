#ifndef SILL_MIXTURE_EM3_HPP
#define SILL_MIXTURE_EM3_HPP

#include <sill/factor/factor_evaluator.hpp>
#include <sill/factor/mixture.hpp>
#include <sill/factor/moment_gaussian.hpp>

#include <iostream>

#include <boost/random/mersenne_twister.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  namespace impl {
    inline void initialize_em(size_t num_components,
                              const vector_domain& args,
                              const vector_dataset<>* ds,
                              const factor_mle<moment_gaussian>::param_type& params,
                              unsigned seed,
                              mixture<moment_gaussian>& mog);
  }

  // the dataset size must not change after initialization
  template <typename F>
  class mixture_em {
  public:
    // Member types of the Learner concept
    typedef double                               real_type; // TODO
    typedef mixture<F>                           model_type;
    typedef typename factor_mle<F>::dataset_type dataset_type;
    typedef typename factor_mle<F>::param_type   comp_param_type;

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
    typedef typename F::domain_type domain_type;

  private:
    typedef typename dataset_type::record_type   record_type;
    typedef arma::Mat<real_type> mat_type;
    typedef arma::Col<real_type> vec_type;

  public:
    //! Creates the learner for the given arguments and number of components
    mixture_em(size_t num_components, const domain_type& args)
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
      impl::initialize_em(num_components, args, dataset, params, seed, model);
    }

    //! Performs one expectation and one maximization step
    //! \return the log-likelihood of the previous model
    real_type iterate() {
      // expectation: probability of each datapoint under the components
      mat_type weights(dataset->size(), model.size());
      for (size_t c = 0; c < model.size(); ++c) {
        factor_evaluator<F> eval(model[c]);
        real_type* w = weights.colptr(c);
        foreach(const record_type& r, dataset->records(eval.arg_vector())) {
          *w++ = eval(r.values);
        }
      }
      vec_type norms = sum(weights, 1);
      for (size_t c = 0; c < model.size(); ++c) {
        weights.col(c) /= norms;
      }
      real_type ll = sum(log(norms));

      // maximization: recomputes the components.
      factor_mle<F> estim(dataset, params);
      for (size_t c = 0; c < model.size(); ++c) {
        model[c] = estim(model.arguments(), weights.col(c));
      }
      model.normalize();

      return ll;
    }

    //! Returns the current estimate
    const mixture<F>& estimate() const {
      return model;
    }

  private:
    // persistent members
    size_t num_components;
    domain_type args;
    
    // iteration-specific members
    const dataset_type* dataset;
    comp_param_type params;
    mixture<F> model;
    
  }; // class mixture_em

  // random initialization for mixtures of Gaussians
  namespace impl {
    inline void initialize_em(size_t num_components,
                              const vector_domain& args,
                              const vector_dataset<>* ds,
                              const factor_mle<moment_gaussian>::param_type& params,
                              unsigned seed,
                              mixture<moment_gaussian>& mog) {
      assert(num_components > 0);
      assert(ds->size() >= num_components);

      // generate the row indices that will determine the centers
      // we want to be absolutely certain they are distinct.
      // otherwise, EM will not compute distinct clusters
      boost::mt19937 rng(seed);
      boost::uniform_int<size_t> uniform(0, ds->size() - 1);
      boost::unordered_set<size_t> random_rows;
      while (random_rows.size() < num_components) {
        random_rows.insert(uniform(rng));
      }

      // compute the covariance and set the random means
      factor_mle<moment_gaussian> estim(ds, params);
      mog = mixture_gaussian(num_components, estim(args));
      size_t k = 0;
      foreach(size_t row, random_rows) {
        mog[k++].mean() = ds->record(row, mog[0].head()).values;
      }
    }
  } // namespace impl

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

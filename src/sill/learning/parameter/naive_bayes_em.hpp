#ifndef SILL_NAIVE_BAYES_EM_HPP
#define SILL_NAIVE_BAYES_EM_HPP

#include <sill/factor/factor_mle_incremental.hpp>
#include <sill/model/naive_bayes.hpp>
#include <sill/learning/parameter/naive_bayes_init.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A class that learns a naive Bayes model when the label variable is not
   * observed. This learner supports datasets with missing values.
   */
  template <typename FeatureF>
  class naive_bayes_em {
  public:
    // Learner concept types
    typedef naive_bayes<FeatureF>           model_type;
    typedef typename FeatureF::real_type    real_type;
    typedef typename FeatureF::dataset_type dataset_type;
    
    // Other types
    typedef typename FeatureF::variable_type   variable_type;
    typedef typename FeatureF::var_vector_type var_vector_type;
    typedef typename factor_mle_incremental<FeatureF>::param_type feature_param_type;

    // Learner concept parameter type
    struct param_type {
      feature_param_type feature_params;
      size_t max_iters;
      real_type tolerance;
      unsigned seed;
      bool verbose;
      param_type(const feature_param_type& params = feature_param_type(),
                 size_t max_iters = 100,
                 real_type tolerance = 1e-6,
                 unsigned seed = 0,
                 bool verbose = false)
        : feature_params(params),
          max_iters(max_iters),
          tolerance(tolerance),
          seed(seed),
          verbose(verbose) { }
    };

    /**
     * Constructs a learner for the given label variable and features.
     * The number of classes is implicitly represented in the label variable.
     */
    naive_bayes_em(finite_variable* label, const var_vector_type& features)
      : label(label), features(features) {
      assert(std::find(features.begin(), features.end(), label) == 
             features.end());
    }

    /**
     * Learns a model using the supplied dataset and default regularization
     * parameters for the prior and feature CPDs.
     * \return the log-likelihood of the training set
     */
    real_type learn(const dataset_type& ds, model_type& model) {
      return learn(ds, param_type(), model);
    }

    /**
     * Learns a model using the supplied dataset and specified regularization
     * parameters for the prior and feature CPDs.
     * \return the log-likelihood of the training set
     */
    real_type learn(const dataset_type& ds,
                    const param_type& params,
                    model_type& model) {
      initialize(&ds, params.feature_params, params.seed);
      //std::cout << "Init: " << this->model << std::endl;
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

    /**
     * Initializes the internal state (dataset, parameters, and the estimate).
     */
    void initialize(const dataset_type* dataset,
                    const feature_param_type& params = feature_param_type(),
                    unsigned seed = 0) {
      this->dataset = dataset;
      this->params = params;
      this->niters = 0;
      sill::initialize(label, features, *dataset, params, seed, this->model);
    }

    void initialize(const dataset_type* dataset,
                    const model_type& model,
                    const feature_param_type& params = feature_param_type()) {
      assert(model.label_var() == label);
      this->dataset = dataset;
      this->model = model;
      this->params = params;
      this->niters = 0;
    }

    /**
     * Performs one iteration of expectation-maximization.
     * \return the lower-bound on the log-likelihood of the previous model
     */
    real_type iterate() {
      typedef typename FeatureF::assignment_type assignment_type;
      typedef typename dataset_type::record_type record_type;
      typedef typename dataset_type::const_record_iterator record_iterator;
      typedef factor_mle_incremental<FeatureF> estimator_type;

      // allocate the feature estimators and data iterators
      std::vector<estimator_type> feature_mle;
      std::vector<record_iterator> feature_it;
      feature_mle.reserve(features.size());
      feature_it.reserve(features.size());
      finite_var_vector label_vec(1, label);
      var_vector_type feature_vec(1); 
      foreach(variable_type* feature, features) {
        feature_vec[0] = feature;
        feature_mle.push_back(estimator_type(feature_vec, label_vec));
        feature_it.push_back(dataset->records(feature_vec).first);
      }

      // expectation: the probability of the labels given each datapoint
      // maximization: accumulate the new prior and the feature CPDs
      assignment_type a;
      real_type bound = 0.0;
      table_factor ptail;
      table_factor new_prior;
      foreach(const record_type& r, dataset->records(features)) {
        r.extract(a);
        model.joint(a, ptail);
        real_type sump = ptail.norm_constant();
        bound += r.weight * std::log(sump);
        ptail *= r.weight / sump;
        new_prior += ptail;
        for (size_t i = 0; i < features.size(); ++i) {
          if (feature_it[i]->count_missing() == 0) {
            feature_mle[i].process(feature_it[i]->values, ptail);
          }
          ++feature_it[i];
        }
      }
      
      // set the parameters of the new model
      assert(new_prior.arguments() == make_domain(label));
      model.set_prior(new_prior.normalize());
      for (size_t i = 0; i < features.size(); ++i) {
        model.add_feature(feature_mle[i].estimate());
      }

      // return the log-likelihood
      ++niters;
      return bound;
    }

    /**
     * Returns the number of iterations since the last time initialize was called
     * or the number of iterations performed in the last time train() was invoked.
     */
    size_t num_iters() const {
      return niters;
    }

  private:
    // persistent members
    finite_variable* label;
    var_vector_type features;
    
    // iteration-specific members
    const dataset_type* dataset;
    feature_param_type params;
    naive_bayes<FeatureF> model;
    size_t niters;

  }; // class naive_bayes_em

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

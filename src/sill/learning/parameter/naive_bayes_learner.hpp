#ifndef SILL_NAIVE_BAYES_LEARNER_HPP
#define SILL_NAIVE_BAYES_LEARNER_HPP

#include <sill/model/naive_bayes.hpp>
#include <sill/learning/factor_mle/factor_mle.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A class that can learn naive Bayes models.
   * Models the Learner concept.
   */
  template <typename FeatureF>
  class naive_bayes_learner {
  public:
    // Learner concept types
    typedef naive_bayes<FeatureF>           model_type;
    typedef typename FeatureF::real_type    real_type;
    typedef typename FeatureF::dataset_type dataset_type;

    struct param_type {
      typedef typename factor_mle<table_factor>::param_type prior_param_type;
      typedef typename factor_mle<FeatureF>::param_type     feature_param_type;
      prior_param_type prior;
      feature_param_type feature;
    };
 
    // Other types
    typedef typename FeatureF::variable_type variable_type;
    typedef typename FeatureF::domain_type   domain_type;

    /**
     * Constructs a learner for the given label variable and features.
     */
    naive_bayes_learner(finite_variable* label, const domain_type& features)
      : label(label), features(features) {
      assert(!features.count(label));
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
    real_type learn(const dataset_type& ds, const param_type& params,
                    model_type& model) {
      factor_mle<table_factor> prior_mle(&ds, params.prior);
      model = naive_bayes<FeatureF>(prior_mle(make_domain(label)));

      factor_mle<FeatureF> feature_mle(&ds, params.feature);
      foreach(variable_type* f, features) {
        model.add_feature(feature_mle(make_domain(f), make_domain(label)));
      }

      return model.log_likelihood(ds);
    }
    
  private:
    finite_variable* label;
    domain_type features;
  };

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

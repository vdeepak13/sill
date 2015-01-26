#ifndef SILL_LOGISTIC_REGRESSION_HPP
#define SILL_LOGISTIC_REGRESSION_HPP

#include <sill/factor/softmax_cpd.hpp>

namespace sill {

  /**
   * A clas that learns a (multinomial) logistic regression model.
   * Models the Learner concept.
   */
  template <typename T = double>
  class logistic_regression {
  public:
    // Learner concept types
    typedef softmax_cpd<T>    model_type;
    typedef T                 real_type;
    typedef hybrid_dataset<T> dataset_type;
    typedef typename factor_mle<softmax_cpd<T> >::param_type param_type;
    
    /**
     * Constructs a learner for the given label variable and features.
     */
    logistic_regression(finite_variable* label,
                        const vector_var_vector& features) 
      : label_(label), features_(features) { }

    /**
     * Learns a model using the supplied dataset and default regularization
     * parameters.
     * \return the log-likelihood of the training set
     */
    real_type learn(const dataset_type& ds, model_type& model) {
      return learn(ds, param_type(), model);
    }

    /**
     * Learns a model using the supplied dataset and regularization parameters.
     * \return the log-likleihood of the training set
     */
    real_type learn(const dataset_type& ds, const param_type& params,
                    model_type& model) {
      factor_mle<softmax_cpd<T> > mle(&ds, params);
      model = mle({label_}, features_);
      return model.log_likelihood(ds);
    }
    
  private:
    finite_variable* label_;
    vector_var_vector features_;

  }; // class logistic_regression

} // namespace sill

#endif

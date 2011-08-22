
#ifndef SILL_MODEL_FUNCTORS_HPP
#define SILL_MODEL_FUNCTORS_HPP

/**
 * \file model_functors.hpp
 *         Functors for evaluating models on datasets w.r.t. different losses.
 *
 * @todo Change this to distribution_functors since I use it for factors too.
 */

#include <sill/macros_def.hpp>

namespace sill {

  //! Functor for computing the log likelihood for a model.
  template <typename ModelType>
  struct model_log_likelihood_functor {

    explicit model_log_likelihood_functor(const ModelType& model,
                                          double base = exp(1.))
      : modelptr(&model), base(base) { }

    double operator()(const typename ModelType::record_type r) const {
      assert(modelptr);
      return modelptr->log_likelihood(r, base);
    }

  private:
    const ModelType* modelptr;
    double base;
  };


  //! Functor for computing the conditional log likelihood log P(Y|X)
  //! from a model P(Y,X).
  template <typename ModelType>
  struct model_conditional_log_likelihood_functor {

    explicit
    model_conditional_log_likelihood_functor
    (const ModelType& model,
     const typename ModelType::domain_type& X,
     double base = exp(1.))
      : modelptr(&model), X(X), base(base) { }

    double operator()(const typename ModelType::record_type r) const {
      assert(modelptr);
      return modelptr->conditional_log_likelihood(r, X, base);
    }

  private:
    const ModelType* modelptr;
    typename ModelType::domain_type X;
    double base;
  };


  //! Functor for computing the per-label accuracy for a model.
  template <typename ModelType>
  struct model_per_label_accuracy_functor {

    //! Constructor for accuracy w.r.t. all arguments of the model.
    explicit model_per_label_accuracy_functor(const ModelType& model)
      : modelptr(&model) { }

    //! Constructor for accuracy of predicting Y given X, where the model
    //! is of P(Y,X).
    explicit
    model_per_label_accuracy_functor(const ModelType& model,
                                     const typename ModelType::domain_type& X)
      : modelptr(&model), X(X) { }

    double operator()(const typename ModelType::record_type r) const {
      assert(modelptr);
      if (X.size() == 0)
        return modelptr->per_label_accuracy(r);
      else
        return modelptr->per_label_accuracy(r, X);
    }

  private:
    const ModelType* modelptr;
    const typename ModelType::domain_type X;
  };


  //! Functor for computing the all-or-nothing accuracy for a model.
  template <typename ModelType>
  struct model_accuracy_functor {

    explicit model_accuracy_functor(const ModelType& model)
      : modelptr(&model) { }

    double operator()(const typename ModelType::record_type r) const {
      assert(modelptr);
      return modelptr->accuracy(r);
    }

  private:
    const ModelType* modelptr;
  };


  //! Functor for computing the mean squared error
  //! (mean over variables) for a model.
  //! Note: This is equivalent to per-label accuracy for finite variables.
  template <typename ModelType>
  struct model_mean_squared_error_functor {

    //! Constructor for mean squared error w.r.t. all arguments of the model.
    explicit model_mean_squared_error_functor(const ModelType& model)
      : modelptr(&model) { }

    //! Constructor for mean squared error of predicting Y given X,
    //! where the model is P(Y,X).
    explicit
    model_mean_squared_error_functor(const ModelType& model,
                                     const typename ModelType::domain_type& X)
      : modelptr(&model), X(X) { }

    double operator()(const typename ModelType::record_type r) const {
      assert(modelptr);
      if (X.size() == 0)
        return modelptr->mean_squared_error(r);
      else
        return modelptr->mean_squared_error(r, X);
    }

  private:
    const ModelType* modelptr;
    typename ModelType::domain_type X;
  };

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_MODEL_FUNCTORS_HPP

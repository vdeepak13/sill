#ifndef SILL_MARKOV_CHAIN_MLE_HPP
#define SILL_MARKOV_CHAIN_MLE_HPP

#include <sill/learning/parameter/factor_mle.hpp>
#include <sill/model/markov_chain.hpp>

namespace sill {

  /**
   * A class that can learn a Markov chain.
   * Models the Learner concept.
   */
  template <typename F>
  class markov_chain_mle {
  public:
    // Learner concept types
    typedef markov_chain<F> model_type;
    typedef typename markov_chain<F>::real_type real_type;
    typedef sequence_dataset<typename F::dataset_type> dataset_type;
    typedef typename factor_mle<F>::param_type param_type;

    // other types
    typedef typename markov_chain<F>::arg_vector_type arg_vector_type;

    /**
     * Constructs a learner for the given set of processes and order.
     */
    explicit markov_chain_mle(const arg_vector_type& args, size_t order = 1)
      : args(args), order(order) { }

    /**
     * Learns a model using the supplied dataset and default regularization
     * parameters for the initial distribution and transition model.
     * \return the log-likelihood of the training set.
     */
    real_type learn(const dataset_type& ds, model_type& model) {
      return learn(ds, param_type(), model);
    }

    /**
     * Learns a model using the supplied dataset and specified regularation
     * parameters for the initial distirbutino and transition model.
     * \return the log-likelihood of the training set.
     */
    real_type learn(const dataset_type& ds, const param_type& params,
                    model_type& model) {
      model = markov_chain<F>(args, order);
      real_type ll = 0;

      // compute the initial distribution
      fixed_view<typename F::dataset_type> fixed = ds.fixed(0);
      factor_mle<F> mle_init(&fixed, params);
      model.initial(mle_init(model.current()));
      ll += model.initial().log_likelihood(fixed);
      
      // compute the transition model
      sliding_view<typename F::dataset_type> sliding = ds.sliding(order);
      factor_mle<F> mle_transition(&sliding, params);
      model.transition(mle_transition(model.next(), model.current()));
      ll += model.transition().log_likelihood(sliding);

      return ll;
    }

  private:
    arg_vector_type args;
    size_t order;

  }; // class markov_chain_mle
  
} // namespace sill

#endif

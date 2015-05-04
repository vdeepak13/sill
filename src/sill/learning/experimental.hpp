namespace sill {
  /*
    Task list:
    0) deprecate dataset2 code and clean up the concepts related to existing classes.
    1) implement Bayesian network sampling using dataset3
    2) implement Bayesian network parameter learning using dataset3.
    3) implement naive Bayes and naive Bayes learning.
       both table factor and hybrid conditionals.
    4) implement timed datasets and hmms / hmm learning.
    4) clean up decomposable
    5) clean up validation.
  */

  /*
    Discrete process learners:
    parameter learning of dynamic Bayesian networks (fully observed case):
    - like learning a BN, but tied parameters
    - sliding view over a subset of variables (child and its parents)
    - when the DBN factor is moment_gaussian, table_factor, hybrid<moment_gaussian>
      etc., can do this efficiently using sliding view of hybrid dataset
    
    parameter learning of HMMs (fully observed case):
    - same as learning DBNs
    - sliding view over two successive time steps to learn the transition model
    - sliding view over individual time steps to learn the observation model
    
    Baum-Welch:
    - for each observation sequence, compute the conditional probability of the
      hidden states, given the observation; this gives us the virtual counts
    - recompute the prior model (normalized sum of the conditional joints)
    - recompute the observation model (sum of observations weighted by the
      posterior probability of the state given the sequence)

    Sparse sequence record:
    - 

  */

  //! Draws samples from the distribution
  template <typename Factor>
  class factor_sampler {
    factor_sampler(const Factor& factor,
                   const argvec_type& head,
                   const argvec_type& tail);

    template <typename Engine>
    void operator()(argument_type& out, Engine& rng);
    
    template <typename Engine>
    void operator()(const argument_type& in, argument_type& out, Engine& rng);
  };

  //! Evaluates the factor
  class factor_evaluator<moment_gaussian> {
    typedef typename Factor::argument_type argument_type;
    typedef typename Factor::result_type   result_type;
    factor_evaluator(const moment_gaussian& mg)
      : cg(mg) { }
    operator()(const argument_type& arg) {
      return cg(arg);
    }
  };

  //! Evaluates the factor
  class factor_evaluator<Factor> {
    typedef typename Factor::argument_type argument_type;
    typedef typename Factor::result_type   result_type;
    factor_evaluator(const Factor& factor)
      : factor(factor) { }
    operator()(const argument_type& args) {
      return factor(args);
    }
    operator()(const assignment_type& args) {
      return factor(args);
    }
  };

  //! generic implementation
  class factor_mle<Factor> {
    typedef ... dataset_type;
    typedef ... param_type;
  private:
    mle(const dataset_type* ds,
        const param_type& params = param_type());
    Factor operator()(const domain_type& dom);
    double log_likelihood(const Factor& factor) const;
  };
  
  //domain_type;
  //argvec_type;

  //! Markov chain model
  template <typename Factor>
  class markov_chain {
    typedef typename Factor::variable_type variable_type;
    typedef discrete_process<variable_type>   process_type;
    typedef sequence_dataset<typename Factor::dataset_type> dataset_type;
    markov_chain(process_type process, size_t order = 1);
    markov_chain(const Factor& initial, const Factor& transition);
    Factor& initial();
    Factor& transition();
  };

  //! Markov chain learning
  template <typename Factor>
  class factor_mle<markov_chain<Factor> > {
    typedef markov_chain<Factor>         result_type;
    typedef timed_dataset<variable_type> dataset_type;
    factor_mle(const timed_dataset<variable_type>* ds,
               const param_type& params = param_type());
    result_type operator()(const domain_type& args) const {
      assert(args.size() == 1);
      size_t order = params.order;
      fixed_view<datset_type> fixed = ds->fixed(0, order - 1);
      sliding_view<dataset_type> sliding = ds->sliding(p, order + 1);
      factor_mle<Factor> initial(&fixed, params.initial);
      factor_mle<Factor> transition(&sliding, params.transition);
      return markov_chain<Factor>(initial(), transition());
    }
  };

  //! 
  template <typename Factor>
  class bayesian_network_parameter_learner {
    typedef bayesian_network<Factor>           model_type;
    typedef typename mle<Factor>::dataset_type dataset_type;
    typedef typename mle<Factor>::param_type   param_type;

    bayesian_network_parameter_learner(bayesian_graph<variable_type>& structure);

    model_type learn(const dataset_type& ds, const param_type& params);

  private:
    bayesian_network<F> prototype;
  };

  template <typename S>
  struct Sampler {
    typedef model_type;
    typedef assignment_type;
    typedef dataset_type;
    
    template <typename Engine>
    assignment_type operator()(Engine& rng);

    void operator()(assignment_type& a, Engine& rng);

    void fill(dataset_type& ds, Engine& rng);
  };

  //! 
  template <typename Factor>
  class bayesian_network_sampler {
    bayesian_network<Factor>* model;
    bayesian_network_sampler(bayesi
  };

  template <typename Factor>
  class chow_liu {
    typedef decomposable<Factor>               model_type;
    typedef typename mle<Factor>::dataset_type dataset_type;
    typedef typename mle<Factor>::param_type   param_type;
    typedef typename Factor::marginal_fn_type  marginal_fn_type;

    chow_liu(const domain_type& args);
    model_type learn(const dataset_type& ds, const param_type& params);
    model_type learn(marginal_fn_type estim);
  };

  template <typename FeatureF>
  class naive_bayes_learner {
    typedef naive_bayes<FeatureF>                model_type;
    typedef typename FeatureF::domain_type       domain_type;
    typedef typename mle<FeatureF>::dataset_type dataset_type;
    naive_bayes_learner(finite_variable* cl, const domain_type& features);
    model_type learn(const dataset_type& ds, const param_type& params);
  };

  template <typename StateF, typename EmitF = StateF>
  class hmm {
    StateF initial;
    StateF transition;
    EmitF  emission;
    typedef mle<EmitF>::dataset_type dataset_type;
    double log_likelihood(const dataset_type& ds); // <<< should this be here???
  };

  template <typename StateF, typename EmitF = StateF>
  class kalman_filter {
    
  };

  template <typename StateF, typename EmitF = StateF>
  class viterbi {
    viterbi(const hmm<StateF,EmitF>* model);
    assignment_type arg_max(const assignment_type& a);
    double likelihood(const assignment_type& a);
  };
      
  class hmm_learner {
    typedef hmm<StateF, EmitF>                model_type;
    typedef typename ...                      domain_type;
    typedef typename mle<EmitF>::dataset_type dataset_type;
    hmm_learner(const domain_type& state, const domain_type& obs);
    model_type learn(const dataset_type& ds, const param_type& params);
    double log_likelihood(const dataset_type& ds, const model_type& model);
  };

  template <typename StateF, typename EmitF = StateF>
  class baum_welch {
    baum_welch(const domain_type& state, const domain_type& obs);
    double log_likelihood(const dataset_type& ds, const model_type& model);
    // computes the log-likelihood under the partial observation by 
    // doing the exact inference
  };

  template <typename Factor>
  class mixture_em {
    mixture_em(size_t num_components);
    struct param_type {
      double tol;
      double maxiter;
      mle<Factor>::param_type 
    };
    mixture<Factor> learn(const dataset_type& ds, const param_type& params);
    double iterate(mixture<Factor>& mixture,
                   const param_type params = param_type());
  };

  template <typename Learner>
  class cross_validation {
    typedef typename Learner::model_type           model_type;
    typedef typename Learner::param_type           param_type;
    typedef typename Learner::dataset_type         dataset_type;
    typedef typename dataset_type::slice_view_type slice_view_type;

    template <typename DS>
    cross_validation(Learner* learner, DS* ds, size_t nfolds);

    template <typename DS>
    cross_validation(Learner* learner, DS* ds, size_t nfolds);

    // todo: figure out the best signatures
    validation_result
    validate(const param_type& params) const;

    validation_result
    validate(const param_type& params, model_type& best) const;

    std::vector<validation_result>
    validate(std::vector<param_type>& params) const;

    std::vector<validation_result>
    validate(std::vector<param_type>& params, model_type& best) const;

  private:
    std::vector<slice_view_type> train;
    std::vector<slice_view_type> test;
  };

  template <typename Learner>
  class validation {
    typedef typename Learner::model_type           model_type;
    typedef typename Learner::param_type           param_type;
    typedef typename Learner::dataset_type         dataset_type;
    typedef typename dataset_type::slice_view_type slice_view_type;
    
    validation(Learner* learner, DS* ds, double fraction_train);
    validation(Learner* learner, const dataset_type& train, const dataset_type& test);

    validation_result
    validate(const param_type& params) const;

    std::vector<validation_result>
    validate(std::vector<param_type>& params) const;

    slice_view_type train;
    slice_view_type test;
  };

}

#ifndef SILL_NAIVE_BAYES_EM_HPP
#define SILL_NAIVE_BAYES_EM_HPP

#include <sill/factor/random/uniform_table_generator.hpp>
#include <sill/model/naive_bayes.hpp>
#include <sill/learning/parameter/em_parameters.hpp>
#include <sill/learning/parameter/naive_bayes_init.hpp>
#include <sill/learning/iterative_parameters.hpp>

#include <cmath>
#include <random>
#include <vector>

namespace sill {

  /**
   * A class that learns a naive Bayes model when the label variable
   * is not observed. The objective value of this algorithm is
   * a lower-bound on the log-likelihood of the model.
   */
  template <typename LabelF, typename FeatureF = LabelF>
  class naive_bayes_em {
    typedef typename FeatureF::mle_type::regul_type regul_type;

  public:
    // Learner concept types
    typedef naive_bayes<LabelF, FeatureF> model_type;
    typedef typename LabelF::real_type    real_type;
    typedef em_parameters<regul_type>     param_type;
    
    // Other types
    typedef typename LabelF::variable_type variable_type;
    typedef std::vector<variable_type>    var_vector_type;

    /**
     * Constructs a naive Bayes learner with given parameters.
     */
    explicit naive_bayes_em(const parm_type& param = param_type())
      : param_(param_) { }

    /**
     * Fits a model using the supplied dataset for the given label variable
     * and feature vector.
     */
    template <typename Dataset>
    naive_bayes_em& fit(const Dataset& ds,
                        variable_type label,
                        const var_vector_type& features) {
      reset(&ds, label, features);
      fit();
    }

    /**
     * Fits a model that was previous initialized using reset().
     */
    naive_bayes_em& fit() {
      while (iteration_ < param_.niter) {
        real_type previous = objective_;
        iterate();
        if (param_.verbose) {
          std::cout << "Iteration " << it << ": ll=" << objective_ << std::endl;
        }
        if (iteration_ > 1 &&
            std::abs(objective_ - previous) / ds.size() < param_.tol) {
          break;
        }
      }
      return *this;
    }

    /**
     * Initializes the model for table-like factors by drawing the
     * parameters uniormly at random and sets the dataset for training.
     */
    template <typename Dataset>
    void reset(const Dataset* ds,
               variable_type label,
               const var_vector_type& features) {
      // initialize the model
      uniform_table_generator<LabelF> prior_gen;
      uniform_table_generator<FeatureF> cpd_gen;
      std::mt19937 rng(param_.seed);
      model_ = model_type(prior_gen({label}, rng).normalize());
      for (variable_type feature : features) {
        model_.add_feature(cpd_gen({feature, label}, rng));
      }
      // initialize the udpater and statistics
      updater_ = updater<Dataset>(ds, &model_, param_.regul);
      objective_ = std::numeric_limits<real_type>::infinity();
      iteration_ = 0;
    }

    /**
     * Initializes the estimate to the given model and sets the
     * dataset for training.
     */
    template <typename Dataset>
    void reset(const Dataset* ds, const model_type& model) {
      model_ = model;
      updater_ = updater<Dataset>(ds, &model_, param_.regul);
      objective_ = std::numeric_limits<real_type>::infinity();
      iteration_ = 0;
    }

    //! Performs one iteration of EM.
    real_type iterate() { ++iteration_; return (objective_ = updater_()); }
    
    //! Returns the estimated model.
    model_type& model() { return model_; }

    //! Returns the objective value.
    real_type objective() const { return objective_; }

    //! Returns the number of iterations.
    size_t iteration() const { return iteration_; }

  private:
    /**
     * A class that performs EM updates. Implementing this functionality here
     * instead of the enclosing class erases the type of the dataset.
     */
    template <typename Dataset>
    class updater {
    public:
      updater(const Dataset* ds, model_type* model, const regul_type& regul)
        : ds_(ds), model_(model), mle_(regul) {
        label_ = model->label();
        features_.assign(model->features().begin(), model->features().end());
      }

      real_type operator()() {
        // initialize the iterators and CPDs over (feature, label) domains
        size_t n = features_.size();
        std::vector<typename Dataset::const_iterator> it(n);
        std::vector<FeatureF> cpd(n);
        for (size_t i = 0; i < n; ++i) {
          it[i]  = (*ds_)({features_[i], label_}).begin();
          cpd[i] = FeatureF({features_[i], label_});
          mle_.initialize(cpd[i].param());
        }
        
        // expectation: the probability of the labels given each datapoint
        // maximization: accumulate the new prior and the feature CPDs
        real_type bound(0);
        LabelF prior(label_, result_type(0));
        LabelF ptail;
        for (const auto& p : ds_->assignments(features)) {
          model_->restrict(p.first, ptail);
          real_type norm = ptail.marginal();
          bound += p.second * std::log(norm);
          ptail *= p.second / norm;
          prior.param() += ptail.param();
          for (size_t i = 0; i < features.size(); ++i) {
            mle_.process(it[i]->first, ptail.param(), cpd.param());
            ++it[i];
          }
        }

        // set the parameters of the new model
        model_->prior(prior.normalize());
        for (size_t i = 0; i < n; ++i) {
          mle_.finalize(cpd[i].param());
          model_->add_feature(cpd[i]);
        }
        
        return bound;
      }

    private:
      const Dataset* ds_;
      model_type* model_;
      variable_type label_;
      std::vector<variable_type> features_;
      typename FeatureF::mle_type mle_;

    }; // class updater

  private:
    model_type model_;
    std::function<real_type()> updater_;
    real_type objective_;
    size_t iteration_;

  }; // class naive_bayes_em

} // namespace sill

#endif

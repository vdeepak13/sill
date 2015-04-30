#ifndef SILL_NAIVE_BAYES_HPP
#define SILL_NAIVE_BAYES_HPP

#include <sill/math/likelihood/range_ll.hpp>

#include <unordered_map>

namespace sill {

  template <typename F>
  class naive_bayes {
  public:
    // Public type declarations
    //==========================================================================
  public:
    // FactorizedModel types
    typedef typename F::real_type       real_type;
    typedef typename F::result_type     result_type;
    typedef typename F::variable_type   variable_type;
    typedef typename F::domain_type     domain_type;
    typedef typename F::assignment_type assignment_type;
    typedef F                           factor_type;

    // Constructors
    //==========================================================================
  public:
    //! Default constructor; creates a naive Bayes with uninitialized prior.
    naive_bayes() { }
    
    //! Creates a naive Bayes model with the given label variable and
    //! uniform prior
    explicit naive_bayes(variable_type* label)
      : prior_({label}, 1.0) {
      check_prior(prior_);
    }

    //! Creates a naive Bayes model with given prior distribution and CPDs.
    explicit naive_bayes(const F& prior,
                         const std::vector<F>& cpds = std::vector<F>())
      : prior_(prior) {
      check_prior(prior_);
      for (const F& cpd : cpds) {
        add_feature(cpd);
      }
    }

    //! Sets the prior. Must not change the label variable if one is set already.
    void set_prior(const F& prior) {
      check_prior(prior);
      if (prior_.arguments().empty() ||
          prior_.arguments() == prior.arguments()) {
        prior_ = prior;
      } else {
        throw std::invalid_argument("attempt to change the label variable");
      }
    }

    /**
     * Adds a new feature or overwrites the existing one.
     * The prior must have already been set.
     */
    void add_feature(const F& cpd) {
      const domain_type& cpd_args = cpd.arguments();
      if (cpd_args.size() != 2 || !cpd_args.count(label_var())) {
        throw std::invalid_argument("naive_bayes::add_feature() must contain the "
                                    "label variable and exactly one other variable");
      }
      variable_type* f = *cpd_args.begin();
      if (f == label_var()) {
        f = *++cpd_args.begin();
      }
      features_[f] = cpd.reorder(domain_type({label_var(), f}));
    }

    // Queries
    //===================================================================
    //! Returns the label variable.
    variable_type* label_var() const {
      if (prior_.arguments().empty()) {
        throw std::runtime_error("The naive_bayes object is uninitialized");
      }
      return *prior_.arguments().begin();
    }

    //! Returns the features in the model.
    domain_type features() const {
      domain_type result;
      for (const auto& p : features_) {
        result.insert(result.end(), p.first);
      }
      return result;
    }

    //! Returns all the variables in the model.
    domain_type arguments() const {
      domain_type result = features();
      result.insert(result.end(), label_var());
      return result;
    }

    //! Returns the prior distribution.
    const F& prior() const {
      return prior_;
    }
    
    //! Returns the feature CPD.
    const F& feature_cpd(variable_type* v) const {
      return features_.at(v);
    }

    //! Returns true if the model contains the given variable.
    bool contains(variable_type* v) const {
      return v == label_var() || features_.count(v);
    }

    //! Returns the prior multiplied by the likelihood of the assignment.
    void joint(const assignment_type& a, F& result) const {
      result = prior_;
      foreach (const auto& p : features_) {
        if (a.count(p.first)) {
          p.second.restrict_multiply(a, result);
        }
      }
    }

    //! Returns the posterior distribution conditioned on an assignment.
    F posterior(const assignment_type& a) const {
      F tmp;
      joint(a, tmp);
      tmp.normalize();
      return tmp;
    }

    //! Returns the probability of an assignment to label and features.
    result_type operator()(const assignment_type& a) const {
      result_type result = prior_(a);
      for (const auto& p : features_) {
        result *= p.second(a);
      }
      return result;
    }

    //! Returns the log-probability of an assignment to label and featuers.
    real_type log(const assignment_type& a) const {
      real_type result = prior_.log(a);
      for (const auto & p : features_) {
        result += p.second.log(a);
      }
      return result;
    }

    //! Returns the complete log-likelihood of a dataset.
    template <typename Dataset>
    typename std::enable_if<is_dataset<Dataset>::value, real_type>::type
    log(const Dataset& ds) const {
      typedef range_ll<typename F::ll_type> ll_type;
      real_type result;
      result = ll_type(prior_.param()).value(ds(prior_.arguments()));
      for (const auto& p : features_) {
        const F& cpd = p.second;
        result += ll_type(cpd.param()).value(ds(cpd.arguments()));
      }
      return result;
    }

    //! Returns the conditional log-likelihood of a dataset.
    template <typename Dataset>
    typename std::enable_if<is_dataset<Dataset>::value, real_type>::type
    real_type conditional_log(const Dataset& ds) const {
      real_type result(0);
      for (const auto& p : ds.assignments(arguments())) {
        result += posterior(p.first).log(p.first) * p.second;
      }
      return result;
    }

    //! Computes the accuracy of the predictions for a dataset.
    template <typename Dataset>
    typename std::enable_if<is_dataset<Dataset>::value, real_type>::type
    accuracy(const dataset_type& ds) const {
      variable_type* label = label_var();
      real_type result(0);
      real_type weight(0);
      assignment_type a;
      F posterior;
      for (const auto& p : ds.assignments(arguments())) {
        joint(p.first, posterior);
        posterior.maximum(a);
        result += p.second * (a.at(label) == p.first[label]);
        weight += p.second;
      }
      return result / weight;
    }

    friend std::ostream& operator<<(std::ostream& out, const naive_bayes& nb) {
      out << "Prior:" << std::endl << nb.prior_ << std::endl;
      out << "CPDs: " << std::endl;
      for (const auto p : nb.features_) {
        out << p.second;
      }
      return out;
    }

  private:
    void check_prior(const F& prior) const {
      if (prior.arguments().size() != 1) {
        throw std::invalid_argument("the prior must have exactly one argument");
      }
    }
    
    F prior_;
    std::unordered_map<variable_type*, F> features_;

  }; // class naive_bayes

} // namespace sill

#endif

#ifndef SILL_NAIVE_BAYES_HPP
#define SILL_NAIVE_BAYES_HPP

#include <sill/boost_unordered_utils.hpp>
#include <sill/base/finite_variable.hpp>
#include <sill/factor/table_factor.hpp>

#include <boost/unordered_map.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  template <typename FeatureF>
  class naive_bayes {
  public:
    typedef FeatureF                           feature_factor_type;
    typedef typename FeatureF::real_type       real_type;
    typedef typename FeatureF::result_type     result_type;
    typedef typename FeatureF::variable_type   variable_type;
    typedef typename FeatureF::domain_type     domain_type;
    typedef typename FeatureF::assignment_type assignment_type;
    typedef typename FeatureF::dataset_type    dataset_type;
    typedef typename dataset_type::record_type record_type;

    // helper typedef until we use C++11
    typedef std::pair<variable_type* const, FeatureF> variable_cpd_pair;
    
    //! Default constructor; creates a naive Bayes with uninitialized prior
    naive_bayes() { }
    
    //! Creates a naive Bayes model with the given label variable and
    //! uniform prior
    explicit naive_bayes(finite_variable* label)
      : prior_(make_domain(label), 1.0) {
      check_prior(prior_);
    }

    //! Creates a naive Bayes model with given prior distribution and CPDs.
    explicit naive_bayes(const table_factor& prior,
                         const std::vector<FeatureF> feature_cpds = 
                         std::vector<FeatureF>())
      : prior_(prior) {
      check_prior(prior_);
      foreach(const FeatureF& cpd, feature_cpds) {
        add_feature(cpd);
      }
    }

    //! Sets the prior. Must not change the label variable if one is set already.
    void set_prior(const table_factor& prior) {
      check_prior(prior);
      if (prior_.arguments().empty() ||
          prior_.arguments() == prior.arguments()) {
        prior_ = prior;
      } else {
        throw std::invalid_argument("attempt to change the label variable");
      }
    }

    //! Adds a new feature or overwrites the existing one.
    //! The prior must have already been set.
    void add_feature(const FeatureF& cpd) {
      const domain_type& cpd_args = cpd.arguments();
      if (cpd_args.size() != 2 || !cpd_args.count(label_var())) {
        throw std::invalid_argument("naive_bayes::add_feature() must contain the "
                                    "label variable and exactly one other variable");
      }
      variable_type* f = *cpd_args.begin();
      if (f == label_var()) {
        f = *++cpd_args.begin();
      }
      features_[f] = cpd.reorder(make_vector(label_var(), f));
    }

    // Queries
    //===================================================================
    //! Returns the label variable
    finite_variable* label_var() const {
      if (prior_.arguments().empty()) {
        throw std::runtime_error("The naive_bayes object is uninitialized");
      }
      return *prior_.arguments().begin();
    }

    //! Returns the features in the model
    domain_type features() const {
      domain_type result;
      foreach (const variable_cpd_pair& p, features_) {
        result.insert(p.first);
      }
      return result;
    }

    //! Returns all the variables in the model
    domain_type arguments() const {
      domain_type result = features();
      result.insert(label_var());
      return result;
    }

    //! Returns the prior distribution
    const table_factor& prior() const {
      return prior_;
    }
    
    //! Returns the feature CPD
    const FeatureF& feature_cpd(variable_type* v) const {
      return get(features_, v);
    }

    //! Returns true if the model contains the given variable
    bool contains(variable_type* v) const {
      return v == label_var() || features_.count(v);
    }

    //! Returns the prior multiplied by the likelihood of the assignment
    void joint(const assignment_type& a, table_factor& result) const {
      result = prior_;
      foreach (const variable_cpd_pair& p, features_) {
        p.second.restrict_multiply(a, result);
      }
    }

    //! Returns the posterior distribution conditioned on an assignment
    table_factor posterior(const assignment_type& a) const {
      table_factor tmp;
      joint(a, tmp);
      return tmp.normalize();
    }

    //! Returns the probability of an assignment to label and features
    result_type operator()(const assignment_type& a) const {
      result_type result = prior_(a);
      foreach (const variable_cpd_pair& p, features_) {
        result *= p.second(a);
      }
      return result;
    }

    //! Returns the complete log-likelihood of a dataset
    real_type log_likelihood(const dataset_type& ds) const {
      real_type result = prior_.log_likelihood(ds);
      foreach (const variable_cpd_pair& p, features_) {
        result += p.second.log_likelihood(ds);
      }
      return result;
    }

    //! Returns the conditional log-likelihood of a dataset
    real_type conditional_log_likelihood(const dataset_type& ds) const {
      real_type result = 0;
      assignment_type a;
      foreach (const record_type& r, ds.records(make_vector(arguments()))) {
        r.extract(a);
        table_factor conditional = posterior(a);
        result += r.weight * std::log(conditional(a));
      }
      return result;
    }

    friend std::ostream& operator<<(std::ostream& out, const naive_bayes& nb) {
      out << "Prior:" << std::endl << nb.prior_ << std::endl;
      out << "CPDs: " << std::endl;
      foreach(const variable_cpd_pair& p, nb.features_) {
        out << p.second << std::endl;
      }
      return out;
    }

  private:
    void check_prior(const table_factor& prior) const {
      if (prior.arguments().size() != 1) {
        throw std::invalid_argument("the prior must have exactly one argument");
      }
    }
    
    table_factor prior_;
    boost::unordered_map<variable_type*, FeatureF> features_;
  }; // class naive_bayes

  // Typedefs of standard models
  typedef naive_bayes<table_factor>                    multinomial_naive_bayes;
  //typedef naive_bayes<hybrid_factor<moment_gaussian> > gaussian_naive_bayes;
    
} // namespace sill

#include <sill/macros_undef.hpp>

#endif

#include <sill/learning/discriminative/multilabel_classifier.hpp>

#include <sill/macros_def.hpp>

namespace sill {

    // Getters and helpers
    //==========================================================================

    std::vector<size_t> multilabel_classifier::nclasses() const {
      std::vector<size_t> v(labels_.size());
      for (size_t j = 0; j < labels_.size(); ++j)
        v[j] = labels_[j]->size();
      return v;
    }

    void multilabel_classifier::label(const dataset<la_type>& ds, size_t i, std::vector<size_t>& v) const {
      v.resize(labels_.size());
      for (size_t j = 0; j < labels_.size(); ++j)
        v[j] = ds.finite(i,label_indices_[j]);
    }

    void multilabel_classifier::label(const dataset<la_type>& ds, size_t i, vec& v) const {
      v.resize(labels_.size());
      for (size_t j = 0; j < labels_.size(); ++j)
        v[j] = ds.finite(i,label_indices_[j]);
    }

    void multilabel_classifier::label(const record_type& r, std::vector<size_t>& v) const {
      v.resize(labels_.size());
      for (size_t j = 0; j < labels_.size(); ++j)
        v[j] = r.finite(label_indices_[j]);
    }

    void multilabel_classifier::label(const record_type& r, vec& v) const {
      v.resize(labels_.size());
      for (size_t j = 0; j < labels_.size(); ++j)
        v[j] = r.finite(label_indices_[j]);
    }

    void multilabel_classifier::label(const assignment& example, std::vector<size_t>& v) const {
      v.resize(labels_.size());
      const finite_assignment& fa = example.finite();
      for (size_t j = 0; j < labels_.size(); ++j)
        v[j] = safe_get(fa, labels_[j]);
    }

    void multilabel_classifier::label(const assignment& example, vec& v) const {
      v.resize(labels_.size());
      const finite_assignment& fa = example.finite();
      for (size_t j = 0; j < labels_.size(); ++j)
        v[j] = safe_get(fa, labels_[j]);
    }
    void
    multilabel_classifier::assign_labels(const std::vector<size_t>& r, finite_assignment& fa) const {
      for (size_t j(0); j < labels_.size(); ++j)
        fa[labels_[j]] = r[label_indices_[j]];
    }

    vec multilabel_classifier::test_accuracy(const dataset<la_type>& testds) const {
      vec test_acc(nlabels(), 0);
      if (testds.size() == 0) {
        std::cerr << "multilabel_classifier::test_accuracy() called with an"
                  << " empty dataset." << std::endl;
        assert(false);
        return test_acc;
      }
      dataset<la_type>::record_iterator testds_end = testds.end();
      std::vector<size_t> truth;
      for (dataset<la_type>::record_iterator testds_it = testds.begin();
           testds_it != testds_end; ++testds_it) {
        const record_type& example = *testds_it;
        std::vector<size_t> pred(predict(example));
        label(example, truth);
        for (size_t j(0); j < labels_.size(); ++j)
          if (pred[j] == truth[j])
            ++test_acc[j];
      }
      for (size_t j(0); j < labels_.size(); ++j)
        test_acc[j] /= testds.size();
      return test_acc;
    }

    std::pair<double, double> multilabel_classifier::test_log_likelihood(const dataset<la_type>& testds,
                                                  double base) const {
      if (testds.size() == 0) {
        std::cerr << "multilabel_classifier::test_log_likelihood() called with"
                  << " an empty dataset." << std::endl;
        assert(false);
        return std::make_pair(0,0);
      }
      double loglike(0);
      double stddev(0);
      dataset<la_type>::record_iterator testds_end = testds.end();
      finite_assignment fa;
      for (dataset<la_type>::record_iterator testds_it = testds.begin();
           testds_it != testds_end; ++testds_it) {
        const record_type& example = *testds_it;
        assign_labels(example.finite(), fa);
        double ll(probabilities(example).v(fa));
        if (ll == 0) {
          loglike = - std::numeric_limits<double>::infinity();
          stddev = std::numeric_limits<double>::infinity();
          break;
        }
        ll = std::log(ll);
        loglike += ll;
        stddev += ll * ll;
      }
      loglike /= std::log(base);
      stddev /= (std::log(base) * std::log(base));
      if (testds.size() == 1)
        return std::make_pair(loglike, std::numeric_limits<double>::infinity());
      stddev = sqrt((stddev - loglike * loglike / testds.size())
                    / (testds.size() - 1));
      loglike /= testds.size();
      return std::make_pair(loglike, stddev);
    }

    // Prediction methods
    //==========================================================================

    std::vector<size_t> multilabel_classifier::predict(const record_type& example) const {
      std::vector<size_t> preds(labels_.size());
      predict(example, preds);
      return preds;
    }

    std::vector<size_t> multilabel_classifier::predict(const assignment& example) const {
      std::vector<size_t> preds(labels_.size());
      predict(example, preds);
      return preds;
    }

    std::vector<vec> multilabel_classifier::confidences(const record_type& example) const {
      std::vector<vec> c(labels_.size());
      std::vector<size_t> preds(predict(example));
      for (size_t j = 0; j < labels_.size(); ++j) {
        c[j].resize(labels_[j]->size(), -1);
        c[j][preds[j]] = 1;
      }
      return c;
    }

    std::vector<vec> multilabel_classifier::confidences(const assignment& example) const {
      std::vector<vec> c(labels_.size());
      std::vector<size_t> preds(predict(example));
      for (size_t j = 0; j < labels_.size(); ++j) {
        c[j].resize(labels_[j]->size(), -1);
        c[j][preds[j]] = 1;
      }
      return c;
    }

    std::vector<vec>
    multilabel_classifier::marginal_probabilities(const record_type& example) const {
      std::vector<vec> c(labels_.size());
      std::vector<size_t> preds(predict(example));
      for (size_t j = 0; j < labels_.size(); ++j) {
        c[j].resize(labels_[j]->size(), 0);
        c[j][preds[j]] = 1;
      }
      return c;
    }

    std::vector<vec>
    multilabel_classifier::marginal_probabilities(const assignment& example) const {
      std::vector<vec> c(labels_.size());
      std::vector<size_t> preds(predict(example));
      for (size_t j = 0; j < labels_.size(); ++j) {
        c[j].resize(labels_[j]->size(), 0);
        c[j][preds[j]] = 1;
      }
      return c;
    }

    table_factor multilabel_classifier::probabilities(const record_type& example) const {
      table_factor f(labels_, 0);
      finite_assignment fa;
      predict(example, fa);
      f.set_v(fa, 1);
      return f;
    }

    table_factor multilabel_classifier::probabilities(const assignment& example) const {
      table_factor f(labels_, 0);
      finite_assignment fa;
      predict(example, fa);
      f.set_v(fa, 1);
      return f;
    }

    // Save and load methods
    //==========================================================================

    void multilabel_classifier::save(std::ofstream& out, size_t save_part,
                      bool save_name) const {
      base::save(out, save_part, save_name);
      out << label_indices_ << "\n";
    }

    bool
    multilabel_classifier::load(std::ifstream& in, const datasource& ds, size_t load_part) {
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      read_vec(is, label_indices_);
      labels_.resize(label_indices_.size(), NULL);
      for (size_t j = 0; j < label_indices_.size(); ++j) {
        if (label_indices_[j] < ds.num_finite())
          labels_[j] = ds.finite_list()[label_indices_[j]];
        else {
          labels_[j] = NULL;
          assert(false);
          return false;
        }
      }
      return true;
    }

} // namespace sill

#include <sill/macros_undef.hpp>

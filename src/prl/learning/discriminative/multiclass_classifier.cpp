#include <prl/learning/discriminative/multiclass_classifier.hpp>

#include <prl/macros_def.hpp>

namespace prl {

    // Getters and helpers
    //==========================================================================

    size_t multiclass_classifier::nclasses() const {
      if (label_ == NULL) {
        assert(false);
        return 0;
      }
      return label_->size();
    }

    std::pair<double, double> multiclass_classifier::test_log_likelihood(const dataset& testds,
                                                  double base) const {
      if (testds.size() == 0) {
        std::cerr << "multiclass_classifier::test_log_likelihood() called with"
                  << " no data." << std::endl;
        assert(false);
        return std::make_pair(0,0);
      }
      double loglike(0);
      double stddev(0);
      dataset::record_iterator testds_end = testds.end();
      for (dataset::record_iterator testds_it = testds.begin();
           testds_it != testds_end; ++testds_it) {
        const record& example = *testds_it;
        double ll(probabilities(example)[label(example)]);
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

    std::pair<double, double> multiclass_classifier::test_log_likelihood(oracle& o, size_t n,
                                                  double base) const {
      size_t cnt(0);
      double loglike(0);
      double stddev(0);
      while (cnt < n) {
        if (!(o.next()))
          break;
        const record& example = o.current();
        double ll(probabilities(example)[label(example)]);
        if (ll == 0) {
          loglike = - std::numeric_limits<double>::infinity();
          stddev = std::numeric_limits<double>::infinity();
          break;
        }
        ll = std::log(ll);
        loglike += ll;
        stddev += ll * ll;
        ++cnt;
      }
      loglike /= std::log(base);
      stddev /= (std::log(base) * std::log(base));
      if (cnt == 0) {
        std::cerr << "multiclass_classifier::test_log_likelihood() called with"
                  << " an oracle with no data."
                  << std::endl;
        assert(false);
        return std::make_pair(0,0);
      } else if (cnt == 1)
        return std::make_pair(loglike, std::numeric_limits<double>::infinity());
      stddev = sqrt((stddev - loglike * loglike / cnt)/(cnt - 1));
      loglike /= cnt;
      return std::make_pair(loglike, stddev);
    }

    // Prediction methods
    //==========================================================================

    vec multiclass_classifier::confidences(const record& example) const {
      vec v(nclasses(), -1);
      size_t j(predict(example));
      v[j] = 1;
      return v;
    }

    vec multiclass_classifier::confidences(const assignment& example) const {
      vec v(nclasses(), -1);
      size_t j(predict(example));
      v[j] = 1;
      return v;
    }

    vec multiclass_classifier::probabilities(const record& example) const {
      vec v(nclasses(), 0);
      size_t j(predict(example));
      v[j] = 1;
      return v;
    }

    vec multiclass_classifier::probabilities(const assignment& example) const {
      vec v(nclasses(), 0);
      size_t j(predict(example));
      v[j] = 1;
      return v;
    }

} // namespace prl

#include <prl/macros_undef.hpp>

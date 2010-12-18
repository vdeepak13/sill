#include <sill/learning/discriminative/classifier_cascade.hpp>
#include <sill/learning/discriminative/load_functions.hpp>

#include <sill/macros_def.hpp>

namespace sill {

    // Parameters: public methods
    //==========================================================================

    void classifier_cascade_parameters::load(std::ifstream& in,
                                             const datasource& ds) {
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      size_t nclassifiers;
      if (!(is >> nclassifiers))
        assert(false);
      if (!(is >> init_base_classifiers))
        assert(false);
      if (!(is >> rare_class))
        assert(false);
      assert(rare_class == 0 || rare_class == 1);
      if (!(is >> max_false_common_rate))
        assert(false);
      assert(max_false_common_rate >= 0 && max_false_common_rate <= 1);
      if (!(is >> base_dataset_size))
        assert(false);
      if (!(is >> random_seed))
        assert(false);
      if (!(is >> max_filter_count))
        assert(false);
      base_classifiers.resize(nclassifiers);
      for (size_t t = 0; t < nclassifiers; ++t)
        base_classifiers[t] = load_binary_classifier(in, ds);
    }

    // Protected methods
    //==========================================================================

    void classifier_cascade::init(const dataset& rare_ds) {
      assert(params.valid());
      assert(base_ds.comparable(common_o));
      rng.seed(static_cast<unsigned>(params.random_seed));
      base_ds.reserve(rare_ds.size());
      for (size_t i(0); i < rare_ds.size(); ++i)
        base_ds.insert(rare_ds[i]);
      params.set_check_params(base_ds.size());
      max_filter_count_ = params.max_filter_count;
      if (base_ds.size() > 0) {
        base_ds.reserve(params.base_dataset_size);
        base_ds_preds.resize(params.base_dataset_size);
        for (size_t t(0); t < params.init_base_classifiers; ++t)
          if (!(step()))
            break;
      }
    }

    bool classifier_cascade::next_example() {
      size_t filter_count(0);
      while (common_o.next()) {
        const record& r = common_o.current();
        if (predict(r) != label(r))
          return true;
        ++filter_count;
        if (filter_count >= max_filter_count_)
          break;
      }
      if (DEBUG_CLASSIFIER_CASCADE)
        std::cerr << "classifier_cascade: exiting from next_example()"
                  << " (called from step()) since"
                  << " max_filter_count was reached.\n"
                  << " (filter_count = " << filter_count
                  << ", max_filter_count = " << max_filter_count_
                  << ")\n"
                  << " If you want to run more iterations,"
                  << " consider increasing the max_filter_count." << std::endl;
      return false;
    }

    // Learning and mutating operations
    //==========================================================================

    bool classifier_cascade::step() {
      if (params.base_classifiers.size() == 0)
        return false;

      // Prepare dataset by adding new common class examples
      if (base_classifiers.size() == 0)
        for (size_t i(rare_ds_size); i < params.base_dataset_size; ++i) {
          if (!(common_o.next())) {
            if (DEBUG_CLASSIFIER_CASCADE)
              std::cerr << "classifier_cascade: exiting from step() since"
                        << " common class oracle is depleted." << std::endl;
            return false;
          }
          base_ds.insert(common_o.current().finite(),
                         common_o.current().vector());
        }
      else
        for (size_t i(rare_ds_size); i < params.base_dataset_size; ++i) {
          if (!next_example())
            return false;
//          std::cerr << "\noracle label = " << label(common_o.current())
//                    << std::endl;
          base_ds.set_record(i, common_o.current().finite(),
                             common_o.current().vector());
//          std::cerr << "record label = " << label(base_ds, i)
//                    << std::endl;
        }

      // Train a new classifier
      statistics stats(base_ds);
      if (base_classifiers.size() >= params.base_classifiers.size()) {
        params.base_classifiers.back()->random_seed
          (boost::uniform_int<int>(0,std::numeric_limits<int>::max())(rng));
        base_classifiers.push_back
          (params.base_classifiers.back()->create(stats));
      } else {
        params.base_classifiers[base_classifiers.size()]->random_seed
          (boost::uniform_int<int>(0,std::numeric_limits<int>::max())(rng));
        base_classifiers.push_back
          (params.base_classifiers[base_classifiers.size()]->create(stats));
      }
      // Choose a threshold
      // (First, store raw predictions in rare_ds_preds and sort them.)
      // TODO: This adjusts the threshold until the rare class is almost
      //       misclassified too often, but it should check to see if the
      //       common class is classified correctly already.
      for (size_t i(0); i < params.base_dataset_size; ++i)
        base_ds_preds[i] = base_classifiers.back()->predict_raw(base_ds[i]);
      std::vector<size_t> sorted_ind(sorted_indices(base_ds_preds));
      if (params.rare_class == 1) {
        // Go from the lowest predictions up until the false negative rate is
        //  too high.
        // False negative rate = n_false_neg / (base_ds.size() - rare_ds_size)
        size_t max_false_neg((size_t)(floor((base_ds.size() - rare_ds_size)
                                            * params.max_false_common_rate)));
        for (size_t i(0); i < base_ds.size(); ++i) {
          if (label(base_ds, sorted_ind[i]) == 1) {
            if (max_false_neg == 0) { // then we have reached the cutoff
              if (i == 0)
                thresholds.push_back(base_ds_preds[sorted_ind[i]]);
              else
                thresholds.push_back(base_ds_preds[sorted_ind[i-1]] -
                                     (base_ds_preds[sorted_ind[i]] -
                                      base_ds_preds[sorted_ind[i-1]]) / 2.);
              break;
            } else
              --max_false_neg;
          }
        }
        if (thresholds.size() < base_classifiers.size())
          thresholds.push_back(base_ds_preds[sorted_ind.back()]);
      } else {  // params.rare_class == 0
        // Go from the highest predictions down until the false positive rate
        //  is too high.
        // False positive rate = n_false_pos / (base_ds.size() - rare_ds_size)
        size_t max_false_pos((size_t)(floor((base_ds.size() - rare_ds_size)
                                            * params.max_false_common_rate)));
        for (size_t i(0); i < base_ds.size(); ++i) {
          size_t j(base_ds.size() - i - 1);
          if (label(base_ds, sorted_ind[j]) == 0) {
            if (max_false_pos == 0) { // then we have reached the cutoff
              if (j == base_ds.size() - 1)
                thresholds.push_back(base_ds_preds[sorted_ind[j]]);
              else
                thresholds.push_back(base_ds_preds[sorted_ind[j]] +
                                     (base_ds_preds[sorted_ind[j+1]] -
                                      base_ds_preds[sorted_ind[j]]) / 2.);
              break;
            } else
              --max_false_pos;
          }
        }
        if (thresholds.size() < base_classifiers.size())
          thresholds.push_back(base_ds_preds[sorted_ind.front()]);
      }
      // Print debugging info
      if (DEBUG_CLASSIFIER_CASCADE) {
        std::cerr << "classifier_cascade: iteration " << base_classifiers.size()
                  << "\n\t Dataset has " << rare_ds_size << " rare examples "
                  << " and " << (base_ds.size() - rare_ds_size)
                  << " common examples.\n"
                  << "\t Base classifier has training accuracy "
                  << base_classifiers.back()->test_accuracy(base_ds) << ".\n"
                  << "\t max_false_pos/neg = "
                  << ((size_t)(floor((base_ds.size() - rare_ds_size)
                                     * params.max_false_common_rate)))
                  << ".\n"
                  << "\t Sorted predictions range: ["
                  << base_ds_preds[sorted_ind.front()] << ", "
                  << base_ds_preds[sorted_ind.back()] << "].\n"
                  << "\t Chosen threshold: " << thresholds.back() << std::endl;
      }

      return true;
    }

    // Prediction methods
    //==========================================================================

    std::size_t classifier_cascade::predict(const assignment& example) const {
      if (params.rare_class == 0) {
        for (size_t t = 0; t < base_classifiers.size(); ++t)
          if (base_classifiers[t]->predict_raw(example) > thresholds[t])
            return 1;
        return 0;
      } else {
        for (size_t t = 0; t < base_classifiers.size(); ++t)
          if (base_classifiers[t]->predict_raw(example) < thresholds[t])
            return 0;
        return 1;
      }
    }

    std::size_t classifier_cascade::predict(const record& example) const {
      if (params.rare_class == 0) {
        for (size_t t = 0; t < base_classifiers.size(); ++t)
          if (base_classifiers[t]->predict_raw(example) > thresholds[t])
            return 1;
        return 0;
      } else {
        for (size_t t = 0; t < base_classifiers.size(); ++t)
          if (base_classifiers[t]->predict_raw(example) < thresholds[t])
            return 0;
        return 1;
      }
    }

    // Save and load methods
    //==========================================================================

    void classifier_cascade::save(std::ofstream& out, size_t save_part,
              bool save_name) const {
      assert(false); // FINISH THIS
      base::save(out, save_part, save_name);
      params.save(out);

      for (size_t t = 0; t < base_classifiers.size(); ++t)
        base_classifiers[t]->save(out);
    }

    bool classifier_cascade::load(std::ifstream& in, const datasource& ds, size_t load_part) {
      assert(false); // FINISH THIS
      if (!(base::load(in, ds, load_part)))
        return false;
      params.load(in, ds);
      std::string line;
      getline(in, line);
      std::istringstream is(line);

      size_t tmpsize;
      if (!(is >> tmpsize))
        assert(false);
      base_classifiers.resize(tmpsize);
      for (size_t t = 0; t < base_classifiers.size(); ++t)
        base_classifiers[t] = load_binary_classifier(in, ds);
      rng.seed(static_cast<unsigned>(params.random_seed));
    }

} // namespace sill

#include <sill/macros_undef.hpp>

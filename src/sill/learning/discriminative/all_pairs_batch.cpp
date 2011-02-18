#include <sill/learning/discriminative/all_pairs_batch.hpp>

#include <sill/macros_def.hpp>

namespace sill {

    // Protected methods
    //==========================================================================

    void all_pairs_batch::init(const datasource& ds) {
      assert(params.valid());
      rng.seed(static_cast<unsigned>(params.random_seed));
      uniform_prob = boost::uniform_real<double>(0,1);
      nclasses_ = nclasses();

      base_classifiers.resize(nclasses_-1);
      base_train_acc.resize(nclasses_-1);
    }

    void all_pairs_batch::build(dataset_statistics& stats) {
      assert(params.binary_label != NULL);
      const dataset& ds = stats.get_dataset();
      init(ds);

      std::vector<double> distrib;
      // indices[k] is a list of indices of records which have label k
      std::vector<std::vector<size_t> > indices(nclasses_);
      for (size_t i = 0; i < ds.size(); i++)
        indices[ds.finite(i,label_index_)].push_back(i);
      // Classify all labels i (0) vs. j (1)
      if (DEBUG_ALL_PAIRS_BATCH)
        std::cerr << "all_pairs (batch): begin training" << std::endl;
      for (size_t i = 0; i < nclasses_ - 1; i++) {
        for (size_t j = i+1; j < nclasses_; j++) {
          // Create a dataset view and distribution
          std::vector<size_t> ind(indices[i]);
          foreach(size_t s, indices[j])
            ind.push_back(s);
//          ind.insert(indices[j].begin(), indices[j].end(), ind.end()); // TODO: What's wrong with this?
          if (ind.size() == 0) {
            base_classifiers[i].push_back
              (boost::shared_ptr<binary_classifier>());
            base_train_acc[i].push_back(.5);
            if (DEBUG_ALL_PAIRS_BATCH)
              std::cerr << "all_pairs (batch): done with (i,j) = ("
                        << i << "," << j << "); training accuracy = "
                        << .5 << std::endl;
            continue;
          }
          dataset_view ds_view(ds, true);
          ds_view.set_binary_indicator(label_, params.binary_label, j);
          ds_view.set_record_indices(ind);
          dataset_statistics stats_view(ds_view);
          params.base_learner->random_seed
            (boost::uniform_int<int>(0, std::numeric_limits<int>::max())(rng));
          if (i == 1 && base_classifiers[i].size() == 5)
            std::cerr << std::endl;
          base_classifiers[i].push_back
            (params.base_learner->create(stats_view));
          base_train_acc[i].push_back
            (base_classifiers[i].back()->train_accuracy());
          if (DEBUG_ALL_PAIRS_BATCH)
            std::cerr << "all_pairs (batch): done with (i,j) = (" << i << ","
                      << j << "); training accuracy = "
                      << base_train_acc[i].back() << std::endl;
        }
      }
    }

    /*
    void all_pairs_batch::build_online(oracle& o) {
      init(o);

      // indices[k] is a list of indices of records which have label k
      std::vector<std::vector<size_t> > indices(nclasses_);
      for (size_t i = 0; i < ds.size(); i++)
        indices[ds.finite(i,label_index_)].push_back(i);
      // Classify all labels i (0) vs. j (1)
      if (DEBUG_ALL_PAIRS_BATCH)
        std::cerr << "all_pairs (batch): begin training" << std::endl;
      for (size_t i = 0; i < nclasses_ - 1; i++) {
        for (size_t j = i+1; j < nclasses_; j++) {
          // Create a dataset view and distribution
          std::vector<size_t> ind(indices[i]);
          foreach(size_t s, indices[j])
            ind.push_back(s);
//          ind.insert(indices[j].begin(), indices[j].end(), ind.end()); // TODO: What's wrong with this?
          if (ind.size() == 0) {
            base_classifiers[i].push_back
              (boost::shared_ptr<binary_classifier>());
            base_train_acc[i].push_back(.5);
            if (DEBUG_ALL_PAIRS_BATCH)
              std::cerr << "all_pairs (batch): done with (i,j) = ("
                        << i << "," << j << "); training accuracy = "
                        << .5 << std::endl;
            continue;
          }
          dataset_view ds_view_
            (ds, label_, params.binary_label, j);
          dataset_view ds_view(ds_view_, ind);
          distrib.clear();
          foreach(size_t k, ind)
            distrib.push_back(ds.weight(k));
          dataset_statistics stats_view(ds_view);
          params.base_learner->random_seed
            (boost::uniform_int<int>(0,std::numeric_limits<int>::max())(rng));
// (fix this)          base_classifiers[i].push_back(params.base_learner->createB(stats_view, distrib));
          base_train_acc[i].push_back
            (base_classifiers[i].back()->train_accuracy());
          if (DEBUG_ALL_PAIRS_BATCH)
            std::cerr << "all_pairs (batch): done with (i,j) = (" << i << "," << j
                      << "); training accuracy = " << base_train_acc[i].back()
                      << std::endl;
        }
      }
    }
    */

    // Prediction methods
    //==========================================================================

    std::size_t all_pairs_batch::predict(const record& example) const {
      std::vector<size_t> wins(nclasses_);
      for (size_t i = 0; i < nclasses_-1; i++) {
        for (size_t j = 0; j < nclasses_-i-1; j++) {
          if (base_classifiers[i][j] == NULL) {
            if (uniform_prob(rng) > .5)
              ++wins[i];
            else
              ++wins[j+i+1];
            continue;
          }
          if (base_classifiers[i][j]->predict(example) == 0)
            ++wins[i];
          else
            ++wins[j+i+1];
        }
      }
      return max_index(wins, rng);
    }

    std::size_t all_pairs_batch::predict(const assignment& example) const {
      std::vector<size_t> wins(nclasses_);
      for (size_t i = 0; i < nclasses_-1; i++) {
        for (size_t j = 0; j < nclasses_-i-1; j++) {
          if (base_classifiers[i][j] == NULL) {
            if (uniform_prob(rng) > .5)
              ++wins[i];
            else
              ++wins[j+i+1];
            continue;
          }
          if (base_classifiers[i][j]->predict(example) == 0)
            ++wins[i];
          else
            ++wins[j+i+1];
        }
      }
      return max_index(wins, rng);
    }

    // Save and load methods
    //==========================================================================

    void all_pairs_batch::save(std::ofstream& out, size_t save_part,
                               bool save_name) const {
      base::save(out, save_part, save_name);
      params.save(out);
      for (size_t i = 0; i < nclasses_-1; ++i)
        for (size_t j = 0; j < nclasses_-i-1; ++j)
          base_classifiers[i][j]->save(out);
      for (size_t i = 0; i < nclasses_-1; ++i)
        out << base_train_acc[i] << " ";
      out << "\n";
    }

    bool all_pairs_batch::load(std::ifstream& in, const datasource& ds,
                               size_t load_part) {
      if (!(base::load(in, ds, load_part)))
        return false;
      nclasses_ = nclasses();
      params.load(in, ds);
      base_classifiers.resize(nclasses_-1);
      for (size_t i = 0; i < nclasses_-1; ++i) {
        base_classifiers[i].resize(nclasses_-i-1);
        for (size_t j = 0; j < nclasses_-i-1; ++i)
          base_classifiers[i][j] = load_binary_classifier(in, ds);
      }
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      base_train_acc.resize(nclasses_-1);
      for (size_t i = 0; i < nclasses_-1; ++i)
        read_vec(is, base_train_acc[i]);
      rng.seed(static_cast<unsigned>(params.random_seed));
      uniform_prob = boost::uniform_real<double>(0,1);
      return true;
    }

} // namespace sill

#include <sill/macros_undef.hpp>

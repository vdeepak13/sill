#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/learning/discriminative/load_functions.hpp>
#include <sill/learning/discriminative/multiclass2multilabel.hpp>

#include <sill/macros_def.hpp>

namespace sill {

    // Parameters: public methods
    //==========================================================================

    void multiclass2multilabel_parameters::load(std::ifstream& in,
                                                const datasource& ds) {
      base_learner = load_multiclass_classifier(in, ds);
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      if (!(is >> random_seed))
        assert(false);
    }

    // Protected methods
    //==========================================================================

    void multiclass2multilabel::init_only(const datasource& ds) {
      params.base_learner = base_learner;
      assert(params.valid());
      size_t new_label_size(1);
      for (size_t j = 0; j < labels_.size(); ++j)
        new_label_size *= labels_[j]->size();
      if (params.new_label == NULL ||
          new_label_size != params.new_label->size()) {
        assert(false);
        return;
      }
      vector_dataset orig_ds(ds.datasource_info());
      dataset_view ds_view(orig_ds, true);
      ds_view.set_merged_variables(orig_ds.finite_class_variables(),
                                   params.new_label);
      ds_light_view = ds_view.create_light_view();
      tmp_rec = ds_view[0];
    }

    void multiclass2multilabel::build(const dataset& orig_ds) {
      assert(params.valid());
      size_t new_label_size(1);
      for (size_t j = 0; j < labels_.size(); ++j)
        new_label_size *= labels_[j]->size();
      if (params.new_label == NULL ||
          new_label_size != params.new_label->size()) {
        assert(false);
        return;
      }
      dataset_view ds_view(orig_ds, true);
      ds_view.set_merged_variables(orig_ds.finite_class_variables(),
                                   params.new_label);
      dataset_statistics stats(ds_view);

      boost::mt11213b rng(static_cast<unsigned>(params.random_seed));
      params.base_learner->random_seed
        (boost::uniform_int<int>(0, std::numeric_limits<int>::max())(rng));
      base_learner = params.base_learner->create(stats);
      ds_light_view = ds_view.create_light_view();
      tmp_rec = ds_view[0];
    }

    // Prediction methods
    //==========================================================================

    void multiclass2multilabel::predict(const record& example, std::vector<size_t>& v) const {
      ds_light_view->convert_record(example, tmp_rec);
      size_t pred(base_learner->predict(tmp_rec));
      ds_light_view->revert_merged_value(pred, v);
    }

    void multiclass2multilabel::predict(const assignment& example, std::vector<size_t>& v) const {
      ds_light_view->convert_assignment(example, tmp_assign);
      ds_light_view->revert_merged_value(base_learner->predict(tmp_assign), v);
    }

    void multiclass2multilabel::predict(const record& example, finite_assignment& a) const {
      ds_light_view->convert_record(example, tmp_rec);
      ds_light_view->revert_merged_value(base_learner->predict(tmp_rec), a);
    }

    void multiclass2multilabel::predict(const assignment& example, finite_assignment& a) const {
      ds_light_view->convert_assignment(example, tmp_assign);
      ds_light_view->revert_merged_value(base_learner->predict(tmp_assign), a);
    }

    std::vector<vec> multiclass2multilabel::marginal_probabilities(const record& example) const {
      table_factor probs(probabilities(example));
      std::vector<vec> v(labels_.size());
      for (size_t j(0); j < labels_.size(); ++j) {
        v[j].resize(labels_[j]->size());
        size_t j2(0);
        foreach(double val, probs.marginal(make_domain(labels_[j])).values()) {
          v[j][j2] = val;
          ++j2;
        }
      }
      return v;
    }

    std::vector<vec> multiclass2multilabel::marginal_probabilities(const assignment& example) const {
      table_factor probs(probabilities(example));
      std::vector<vec> v(labels_.size());
      for (size_t j(0); j < labels_.size(); ++j) {
        v[j].resize(labels_[j]->size());
        size_t j2(0);
        foreach(double val, probs.marginal(make_domain(labels_[j])).values()) {
          v[j][j2] = val;
          ++j2;
        }
      }
      return v;
    }

    table_factor multiclass2multilabel::probabilities(const record& example) const {
      ds_light_view->convert_record(example, tmp_rec);
      return make_dense_table_factor(labels_,
                                     base_learner->probabilities(tmp_rec));
    }

    table_factor multiclass2multilabel::probabilities(const assignment& example) const {
      ds_light_view->convert_assignment(example, tmp_assign);
      return make_dense_table_factor(labels_,
                                     base_learner->probabilities(tmp_assign));
    }

    // Save and load methods
    //==========================================================================

    void multiclass2multilabel::save(std::ofstream& out, size_t save_part,
              bool save_name) const {
      base::save(out, save_part, save_name);
      params.save(out);
      ds_light_view->save(out);
      base_learner->save(out, 0, true);
    }

    bool multiclass2multilabel::load(std::ifstream& in, const datasource& ds, size_t load_part) {
      if (!(base::load(in, ds, load_part)))
        return false;
      params.load(in, ds);
      ds_light_view->load(in, NULL, params.new_label);
      base_learner = load_multiclass_classifier(in, ds);
      tmp_rec = record(ds.finite_numbering_ptr(), ds.vector_numbering_ptr(),
                       ds.vector_dim());
      return true;
    }

} // namespace sill

#include <sill/macros_undef.hpp>

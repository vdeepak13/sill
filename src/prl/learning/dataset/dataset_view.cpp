#include <sill/learning/dataset/dataset_view.hpp>

#include <sill/macros_def.hpp>

namespace sill {

    // Protected functions
    //==========================================================================

    size_t dataset_view::convert_index(size_t i) const {
      switch(record_view) {
      case RECORD_RANGE:
        return record_min + i;
      case RECORD_INDICES:
        return record_indices[i];
      case RECORD_ALL:
        return i;
      default:
        assert(false);
        return 0;
      }
    }

    // Protected functions required by record
    //==========================================================================

    void dataset_view::convert_assignment_(assignment& a) const {
      finite_assignment& fa = a.finite();
      if (binarized_var != NULL) {
        fa[binary_var] =
          binary_coloring[fa[binarized_var]];
        fa.erase(binarized_var);
      }
      if (m_new_var != NULL) {
        size_t val(0);
        for (size_t j(0); j < m_orig_vars.size(); ++j)
          val += m_multipliers[j] * fa[m_orig_vars[j]];
        fa[m_new_var] = val;
        for (size_t j(0); j < m_orig_vars.size(); ++j)
          fa.erase(m_orig_vars[j]);
      }
      if (vv_view != VAR_ALL) {
        size_t j2(0); // index into vv_finite_var_indices
        for (size_t j(0); j < ds.num_finite(); ++j) {
          if (j2 < vv_finite_var_indices.size() &&
              vv_finite_var_indices[j2] == j) {
            ++j2;
            continue;
          }
          fa.erase(ds.finite_seq[j]);
        }
        vector_assignment& va = a.vector();
        j2 = 0; // index into vv_vector_var_indices
        for (size_t j(0); j < ds.num_vector(); ++j) {
          if (j2 < vv_vector_var_indices.size() &&
              (size_t)(vv_vector_var_indices[j2]) == j) {
            ++j2;
            continue;
          }
          va.erase(ds.vector_seq[j]);
        }
      }
    }

    void dataset_view::load_assignment(size_t i, sill::assignment& a) const {
      ds.load_assignment(convert_index(i), a);
      convert_assignment_(a);
    }

    void dataset_view::load_record(size_t i, record& r) const {
      if (m_new_var != NULL) {
        ds.load_vector(convert_index(i), r.vector());
        ds.load_finite(convert_index(i), tmp_findata);
        // Note: tmp_findata is larger than r.finite() is.
        std::vector<size_t>& fin = r.finite();
        size_t val(0);
        size_t j2(0); // index into m_multipliers
        for (size_t j(0); j < m_orig2new_indices.size(); ++j) {
          if (m_orig2new_indices[j] == std::numeric_limits<size_t>::max()) {
            val += m_multipliers[j2] * tmp_findata[j];
            ++j2;
          } else
            fin[m_orig2new_indices[j]] = tmp_findata[j];
        }
        fin[m_new_var_index] = val;
      } else if (vv_view != VAR_ALL) { // TO DO: SPEED THIS UP!
        ds.load_vector(convert_index(i), tmp_vecdata);
        ds.load_finite(convert_index(i), tmp_findata);
        std::vector<size_t>& fin = r.finite();
        for (size_t j(0); j < vv_finite_var_indices.size(); ++j)
          fin[j] = tmp_findata[vv_finite_var_indices[j]];
        r.vector() = tmp_vecdata(vv_vector_var_indices);
//        vec& v = r.vector();
//        for (size_t j(0); j < vv_vector_var_indices.size(); ++j)
//          v[j] = tmp_vecdata[vv_vector_var_indices[j]];
      } else {
        ds.load_record(convert_index(i), r);
      }
      if (binarized_var != NULL) // move this to below if-then
        r.finite(binarized_var_index) =
          binary_coloring[r.finite(binarized_var_index)];
      r.finite_numbering_ptr = finite_numbering_ptr_;
      r.vector_numbering_ptr = vector_numbering_ptr_;
    }

    void
    dataset_view::load_finite(size_t i, std::vector<size_t>& findata) const {
      if (m_new_var != NULL) {
        ds.load_finite(convert_index(i), tmp_findata);
        // Note: tmp_findata is larger than findata is.
        size_t val(0);
        size_t j2(0); // index into m_multipliers
        for (size_t j(0); j < m_orig2new_indices.size(); ++j) {
          if (m_orig2new_indices[j] ==std::numeric_limits<size_t>::max()){
            val += m_multipliers[j2] * tmp_findata[j];
            ++j2;
          } else
            findata[m_orig2new_indices[j]] = tmp_findata[j];
        }
        findata[m_new_var_index] = val;
      } else if (vv_view != VAR_ALL) {
        ds.load_finite(convert_index(i), tmp_findata);
        for (size_t j(0); j < vv_finite_var_indices.size(); ++j)
          findata[j] = tmp_findata[vv_finite_var_indices[j]];
      } else {
        ds.load_finite(convert_index(i), findata);
      }
      if (binarized_var != NULL) {
        findata[binarized_var_index] =
          binary_coloring[findata[binarized_var_index]];
      }
    }

    void dataset_view::load_vector(size_t i, vec& vecdata) const {
      if (vv_view != VAR_ALL) {
        ds.load_vector(convert_index(i), tmp_vecdata);
        vecdata = tmp_vecdata(vv_vector_var_indices);
//        for (size_t j(0); j < vv_vector_var_indices.size(); ++j)
//          vecdata[j] = tmp_vecdata[vv_vector_var_indices[j]];
      } else
        ds.load_vector(convert_index(i), vecdata);
    }

    void dataset_view::load_assignment_pointer(size_t i, assignment** a) const {
      assert(false);
    }

    // Getters and helpers
    //==========================================================================

    size_t dataset_view::finite(size_t i, size_t j) const {
      size_t val(0);
      if (m_new_var == NULL)
        val = ds.finite(convert_index(i), j);
      else if (vv_view != VAR_ALL) {
        if (j >= finite_seq.size()) {
          assert(false);
          return 0;
        }
        val = ds.finite(convert_index(i), vv_finite_var_indices[j]);
      } else {
        if (j >= finite_seq.size()) {
          assert(false);
          return 0;
        }
        if (j != m_new_var_index)
          val = ds.finite(convert_index(i), m_new2orig_indices[j]);
        else
          for (size_t j(0); j < m_orig_vars.size(); ++j)
            val += m_multipliers[j]
              * ds.finite(convert_index(i), m_orig_vars_indices[j]);
      }
      if (binarized_var == NULL || j != binarized_var_index)
        return val;
      else
        return binary_coloring[val];
    }

    double dataset_view::vector(size_t i, size_t j) const {
      if (vv_view == VAR_INDICES) {
        if (j >= vector_seq.size()) {
          assert(false);
          return 0;
        }
        return ds.vector(convert_index(i), vv_vector_var_indices[j]);
      } else
        return ds.vector(convert_index(i), j);
    }

    void dataset_view::convert_assignment(const assignment& orig_r, assignment& new_r) const {
      new_r.finite() = orig_r.finite();
      new_r.vector() = orig_r.vector();
      /*
      foreach(finite_variable* v, keys(orig_r.finite()))
        new_r.finite()[v] = safe_get(orig_r.finite(), v);
      foreach(vector_variable* v, keys(orig_r.vector()))
        new_r.vector()[v] = safe_get(orig_r.vector(), v);
      */
      convert_assignment_(new_r);
    }

  void dataset_view::convert_record(const record& orig_r, record& new_r) const {
    if (m_new_var != NULL) {
      new_r.vector() = orig_r.vector();
      tmp_findata = orig_r.finite();
      // Note: tmp_findata is larger than new_r.finite() is.
      std::vector<size_t>& fin = new_r.finite();
      size_t val(0);
      size_t j2(0); // index into m_multipliers
      for (size_t j(0); j < m_orig2new_indices.size(); ++j) {
        if (m_orig2new_indices[j] == std::numeric_limits<size_t>::max()) {
          val += m_multipliers[j2] * tmp_findata[j];
          ++j2;
        } else
          fin[m_orig2new_indices[j]] = tmp_findata[j];
      }
      fin[m_new_var_index] = val;
    } else if (vv_view != VAR_ALL) {
      tmp_findata = orig_r.finite();
      tmp_vecdata = orig_r.vector();
      std::vector<size_t>& fin = new_r.finite();
      vec& v = new_r.vector();
      for (size_t j(0); j < vv_finite_var_indices.size(); ++j)
        fin[j] = tmp_findata[vv_finite_var_indices[j]];
      v = tmp_vecdata(vv_vector_var_indices);
//      for (size_t j(0); j < vv_vector_var_indices.size(); ++j)
//        v[j] = tmp_vecdata[vv_vector_var_indices[j]];
    } else {
      new_r = orig_r;
    }
    if (binarized_var != NULL) // move this to below if-then
      new_r.finite(binarized_var_index) =
        binary_coloring[new_r.finite(binarized_var_index)];
    new_r.finite_numbering_ptr = finite_numbering_ptr_;
    new_r.vector_numbering_ptr = vector_numbering_ptr_;
  }

    void dataset_view::revert_merged_value(size_t merged_val,
                             std::vector<size_t>& orig_vals) const {
      size_t val(merged_val);
      for (size_t j(m_orig_vars_sorted.size()-1); j > 0; --j) {
        orig_vals[j] = val / m_multipliers_sorted[j];
        val = val % m_multipliers_sorted[j];
      }
      orig_vals[0] = val;
    }

    void dataset_view::revert_merged_value(size_t merged_val,
                             finite_assignment& orig_vals) const {
      size_t val(merged_val);
      for (size_t j(m_orig_vars_sorted.size()-1); j > 0; --j) {
        orig_vals[m_orig_vars_sorted[j]] = val / m_multipliers_sorted[j];
        val = val % m_multipliers_sorted[j];
      }
      orig_vals[m_orig_vars_sorted[0]] = val;
    }

    boost::shared_ptr<dataset_view> dataset_view::create_light_view() const {
      vector_dataset* tmp_ds_ptr = new vector_dataset(ds.datasource_info());
      boost::shared_ptr<dataset_view>
        view_ptr(new dataset_view(tmp_ds_ptr, *this));
      view_ptr->record_view = record_view;
      view_ptr->record_min = record_min;
      view_ptr->record_max = record_max;
      view_ptr->record_indices = record_indices;
      view_ptr->vv_view = vv_view;
      view_ptr->vv_finite_var_indices = vv_finite_var_indices;
      view_ptr->vv_vector_var_indices = vv_vector_var_indices;
      view_ptr->binarized_var = binarized_var;
      view_ptr->binarized_var_index = binarized_var_index;
      view_ptr->binary_var = binary_var;
      view_ptr->binary_coloring = binary_coloring;
      view_ptr->m_new_var = m_new_var;
      view_ptr->m_new_var_index = m_new_var_index;
      view_ptr->m_orig_vars = m_orig_vars;
      view_ptr->m_orig_vars_indices = m_orig_vars_indices;
      view_ptr->m_multipliers = m_multipliers;
      view_ptr->m_orig_vars_sorted = m_orig_vars_sorted;
      view_ptr->m_multipliers_sorted = m_multipliers_sorted;
      view_ptr->m_orig2new_indices = m_orig2new_indices;
      view_ptr->m_new2orig_indices = m_new2orig_indices;
      view_ptr->tmp_findata = tmp_findata;
      view_ptr->tmp_vecdata = tmp_vecdata;
      return view_ptr;
    }

    // Mutating operations: creating views
    //==========================================================================

    void dataset_view::set_record_range(size_t min, size_t max) {
      assert(min <= max);
      assert(max <= size());
      nrecords = max - min;
      if (record_view == RECORD_RANGE) {
        record_view = RECORD_RANGE;
        record_min = record_min + min;
        record_max = record_min + max;
      } else if (record_view == RECORD_INDICES) {
        record_view = RECORD_INDICES;
        std::vector<size_t> tmp_record_indices;
        for (size_t i = min; i < max; i++)
          tmp_record_indices.push_back(record_indices[i]);
        record_indices = tmp_record_indices;
      } else if (record_view == RECORD_ALL) {
        record_view = RECORD_RANGE;
        record_min = min;
        record_max = max;
      } else
        assert(false);
      if (weighted) {
        vec tmp_weights(max-min);
        for (size_t i = 0; i < max-min; ++i)
          tmp_weights[i] = weights_[min + i];
        weights_ = tmp_weights;
        //        weights_ = weights_.middle(min, max-min);
        // TODO: WHY DOES THE ABOVE LINE NOT WORK?  FIX vector.hpp
      }
    }

    void dataset_view::set_record_indices(const std::vector<size_t>& indices) {
      if (record_view == RECORD_RANGE) {
        record_indices.clear();
        foreach(size_t i, indices) {
          assert(i <= size());
          record_indices.push_back(record_min + i);
        }
      } else if (record_view == RECORD_INDICES) {
        std::vector<size_t> tmp_record_indices;
        foreach(size_t i, indices) {
          assert(i <= size());
          tmp_record_indices.push_back(record_indices[i]);
        }
        record_indices = tmp_record_indices;
      } else if (record_view == RECORD_ALL) {
        record_indices.clear();
        foreach(size_t i, indices) {
          assert(i <= size());
          record_indices.push_back(i);
        }
      } else
        assert(false);
      nrecords = record_indices.size();
      record_view = RECORD_INDICES;
      if (weighted) {
        itpp::ivec tmp_ind(indices.size());
        for (size_t i = 0; i < indices.size(); ++i)
          tmp_ind[i] = indices[i];
        weights_ = weights_(tmp_ind);
      }
    }

  void dataset_view::set_cross_validation_fold(size_t fold, size_t nfolds,
                                               bool heldout) {
    assert(fold < nfolds);
    assert((nfolds > 1) && (nfolds <= nrecords));
    size_t lower((size_t)(floor(fold*nrecords / (double)(nfolds))));
    size_t upper((size_t)(floor((fold+1)*nrecords / (double)(nfolds))));
    if (heldout) {
      set_record_range(lower, upper);
    } else {
      std::vector<size_t> indices;
      for (size_t i(0); i < lower; ++i)
        indices.push_back(i);
      for (size_t i(upper); i < nrecords; ++i)
        indices.push_back(i);
      set_record_indices(indices);
    }
  }

  void dataset_view::save_record_view() {
    assert(saved_record_indices == NULL);
    assert(!weighted);
    saved_record_indices = new std::vector<size_t>();
    switch(record_view) {
    case RECORD_RANGE:
      for (size_t i(record_min); i < record_max; ++i)
        saved_record_indices->push_back(i);
      break;
    case RECORD_INDICES:
      saved_record_indices->operator=(record_indices);
      break;
    case RECORD_ALL:
      for (size_t i(0); i < ds.size(); ++i)
        saved_record_indices->push_back(i);
      break;
    default:
      assert(false);
    }
  }

  void dataset_view::restore_record_view() {
    assert(saved_record_indices);
    assert(!weighted);
    record_view = RECORD_INDICES;
    record_indices = *saved_record_indices;
    nrecords = record_indices.size();
  }

  void
  dataset_view::set_variable_indices(const std::set<size_t>& finite_indices,
                                     const std::set<size_t>& vector_indices) {
    if (binarized_var != NULL || m_new_var != NULL) {
      std::cerr << "dataset_view does not support variable views"
                << " simultaneously with binarized and merged variables yet!"
                << std::endl;
      assert(false);
      return;
    }
    // Check indices
    foreach(size_t j, finite_indices) {
      if (j >= num_finite()) {
        std::cerr << "dataset_view::set_variable_indices() was given finite"
                  << " variable index " << j << ", but there are only "
                  << num_finite() << " finite variables." << std::endl;
        assert(false);
      }
    }
    foreach(size_t j, vector_indices) {
      if (j >= num_vector()) {
        std::cerr << "dataset_view::set_variable_indices() was given vector"
                  << " variable index " << j << ", but there are only "
                  << num_vector() << " vector variables." << std::endl;
        assert(false);
      }
    }
    // Construct view
    finite_var_vector new_finite_class_vars;
    vector_var_vector new_vector_class_vars;
    finite_var_vector vv_finite_vars; // Finite variables in view
    vector_var_vector vv_vector_vars; // Vector variables in view
    if (vv_view == VAR_ALL) {
      std::set<finite_variable*>
        old_finite_class_vars(finite_class_vars.begin(),
                              finite_class_vars.end());
      vv_finite_var_indices.clear();
      for (size_t j = 0; j < num_finite(); ++j)
        if (finite_indices.count(j)) {
          vv_finite_vars.push_back(finite_seq[j]);
          vv_finite_var_indices.push_back(j);
          if (old_finite_class_vars.count(finite_seq[j]))
            new_finite_class_vars.push_back(finite_seq[j]);
        }
      std::set<vector_variable*>
        old_vector_class_vars(vector_class_vars.begin(),
                              vector_class_vars.end());
      vv_vector_var_indices.resize(vector_indices.size());
      size_t j2(0); // index into vv_vector_var_indices
      for (size_t j = 0; j < num_vector(); ++j) {
        if (vector_indices.count(j)) {
          vv_vector_vars.push_back(vector_seq[j]);
          vv_vector_var_indices[j2] = j;
          if (old_vector_class_vars.count(vector_seq[j]))
            new_vector_class_vars.push_back(vector_seq[j]);
          ++j2;
        }
      }
    } else {
      // TODO: IMPLEMENT THIS
      assert(false);
    }
    vv_view = VAR_INDICES;
    std::vector<variable::variable_typenames> new_var_type_order;
    size_t j_f = 0, j_v = 0;
    for (size_t j = 0; j < var_type_order.size(); ++j) {
      if (var_type_order[j] == variable::FINITE_VARIABLE) {
        if (finite_indices.count(j_f))
          new_var_type_order.push_back(variable::FINITE_VARIABLE);
        ++j_f;
      } else {
        if (vector_indices.count(j_v))
          new_var_type_order.push_back(variable::VECTOR_VARIABLE);
        ++j_v;
      }
    }
    // Update datasource info
    finite_vars = finite_domain(vv_finite_vars.begin(), vv_finite_vars.end());
    finite_seq = vv_finite_vars;
    finite_numbering_ptr_->clear();
    dfinite = 0;
    for (size_t j = 0; j < vv_finite_vars.size(); ++j) {
      finite_numbering_ptr_->operator[](vv_finite_vars[j]) = j;
      dfinite += vv_finite_vars[j]->size();
    }
    finite_class_vars = new_finite_class_vars;
    vector_vars = vector_domain(vv_vector_vars.begin(), vv_vector_vars.end());
    vector_seq = vv_vector_vars;
    vector_numbering_ptr_->clear();
    dvector = 0;
    for (size_t j = 0; j < vv_vector_vars.size(); ++j) {
      vector_numbering_ptr_->operator[](vv_vector_vars[j]) = j;
      dvector += vv_vector_vars[j]->size();
    }
    vector_class_vars = new_vector_class_vars;
    var_type_order = new_var_type_order;
    tmp_findata.resize(ds.finite_vars.size());
    tmp_vecdata.resize(ds.dvector);
  }

  void dataset_view::set_variables(const finite_domain& fvars,
                                   const vector_domain& vvars) {
    std::set<size_t> finite_indices;
    std::set<size_t> vector_indices;
    for (size_t j(0); j < finite_seq.size(); ++j)
      if (fvars.count(finite_seq[j]) != 0)
        finite_indices.insert(j);
    for (size_t j(0); j < vector_seq.size(); ++j)
      if (vvars.count(vector_seq[j]) != 0)
        vector_indices.insert(j);
    set_variable_indices(finite_indices, vector_indices);
  }

  void dataset_view::set_binary_coloring(finite_variable* original,
                                         finite_variable* binary,
                                         std::vector<size_t> coloring) {
    if (vv_view != VAR_ALL) {
        std::cerr << "dataset_view does not support binarized variables"
                  << " simultaneously with variable views yet!" << std::endl;
        assert(false);
        return;
      }
      if (binarized_var != NULL) {
        std::cerr << "dataset_view does not support multiple binarized"
                  << " variables yet!" << std::endl;
        assert(false);
        return;
      }
      assert(binary != NULL && binary->size() == 2);
      assert(original != NULL && coloring.size() == original->size());
      assert(finite_vars.count(original));
      for (size_t j = 0; j < coloring.size(); ++j)
        assert(coloring[j] == 0 || coloring[j] == 1);

      binarized_var = original;
      binary_var = binary;
      binary_coloring = coloring;
      binarized_var_index = ds.record_index(original);
      finite_vars.erase(original);
      for (size_t j = 0; j < finite_seq.size(); j++)
        if (finite_seq[j] == original) {
          finite_seq[j] = binary;
          finite_numbering_ptr_->erase(original);
          finite_numbering_ptr_->operator[](binary) = j;
        }
      for (size_t j = 0; j < finite_class_vars.size(); j++)
        if (finite_class_vars[j] == original)
          finite_class_vars[j] = binary;
      dfinite = dfinite - original->size() + 2;
    }

    void dataset_view::set_binary_indicator(finite_variable* original,
                              finite_variable* binary, size_t one_val) {
      assert(original != NULL && one_val < original->size());
      std::vector<size_t> coloring(original->size(), 0);
      coloring[one_val] = 1;
      set_binary_coloring(original, binary, coloring);
    }

    void dataset_view::set_merged_variables(finite_var_vector original_vars,
                              finite_variable* new_var) {
      // Check input.
      if (m_new_var != NULL) {
        std::cerr << "dataset_view::set_merged_variables() may not be called"
                  << " on a view which already has merged variables."
                  << std::endl;
        assert(false);
        return;
      }
      if (vv_view != VAR_ALL) {
        std::cerr << "dataset_view does not support merged variables"
                  << " simultaneously with variable views yet!" << std::endl;
        assert(false);
        return;
      }
      assert(new_var != NULL && !(finite_vars.count(new_var)));
      assert(original_vars.size() > 0);
      size_t new_size(1);
      std::set<finite_variable*> tmp_fin_class_vars(finite_class_vars.begin(),
                                               finite_class_vars.end());
      bool is_class = tmp_fin_class_vars.count(original_vars[0]);
      for (size_t j(0); j < original_vars.size(); ++j) {
        assert(original_vars[j] != NULL &&
               finite_vars.count(original_vars[j]) &&
               original_vars[j] != binarized_var);
        new_size *= original_vars[j]->size();
        if (is_class) {
          if (!tmp_fin_class_vars.count(original_vars[j])) {
            assert(false);
            return;
          }
        } else {
          if (tmp_fin_class_vars.count(original_vars[j])) {
            assert(false);
            return;
          }
        }
      }
      if (new_var->size() != new_size) {
        std::cerr << "dataset_view::set_merged_variables() was given a new "
                  << "finite variable of size " << new_var->size()
                  << " but should have received one of size " << new_size
                  << std::endl;
        assert(false);
        return;
      }
      // Construct the new finite variable ordering, putting the new variable
      //  at the end of the ordering.
      m_new_var = new_var;
      m_new_var_index = finite_vars.size() - original_vars.size();
      m_orig_vars_sorted = original_vars;
      m_orig_vars_indices.clear();
      m_orig2new_indices.clear();
      m_orig2new_indices.resize(finite_vars.size(), 0);
      m_new2orig_indices.resize(finite_vars.size() - original_vars.size() + 1);
      m_multipliers_sorted.resize(original_vars.size());
      tmp_findata.resize(finite_vars.size());
      for (size_t j(0); j < original_vars.size(); ++j) {
        m_orig2new_indices[ds.record_index(original_vars[j])]
          = std::numeric_limits<size_t>::max();
        if (j == 0)
          m_multipliers_sorted[0] = 1;
        else
          m_multipliers_sorted[j] = original_vars[j-1]->size() * m_multipliers_sorted[j-1];
        m_orig_vars_indices.push_back(ds.record_index(original_vars[j]));
      }
      size_t j2(0); // index in new findata corresponding to j
      for (size_t j(0); j < finite_vars.size(); ++j) {
        if (m_orig2new_indices[j] != std::numeric_limits<size_t>::max()) {
          m_orig2new_indices[j] = j2;
          m_new2orig_indices[j2] = j;
          ++j2;
        }
      }
      m_new2orig_indices.back() = std::numeric_limits<size_t>::max();
      assert(m_orig_vars_sorted.size() == original_vars.size()); // check uniqueness
      // Fix indices for binarizing variables.
      if (binarized_var != NULL)
        binarized_var_index = m_orig2new_indices[binarized_var_index];
      // Fix datasource finite variable and variable ordering info.
      for (size_t j(0); j < original_vars.size(); ++j) {
        finite_vars.erase(original_vars[j]);
        dfinite -= original_vars[j]->size();
      }
      finite_vars.insert(new_var);
      dfinite += new_var->size();
      if (is_class) {
        finite_class_vars.clear();
        for (size_t j(0); j < original_vars.size(); ++j)
          tmp_fin_class_vars.erase(original_vars[j]);
        foreach(finite_variable* f, tmp_fin_class_vars)
          finite_class_vars.push_back(f);
        finite_class_vars.push_back(new_var);
      }
      finite_var_vector tmp_finite_seq;
      for (size_t j(0); j < m_new2orig_indices.size() - 1; ++j) {
        tmp_finite_seq.push_back(finite_seq[m_new2orig_indices[j]]);
      }
      tmp_finite_seq.push_back(new_var);
      std::vector<variable::variable_typenames> tmp_var_type_order;
      j2 = 0; // index into original finite_seq
      std::set<size_t> tmp_orig_vars_indices_set(m_orig_vars_indices.begin(),
                                            m_orig_vars_indices.end());
      for (size_t j(0); j < var_type_order.size(); ++j) {
        if (var_type_order[j] == variable::FINITE_VARIABLE) {
          if (!tmp_orig_vars_indices_set.count(j2)) // if not in merged vars
            tmp_var_type_order.push_back(variable::FINITE_VARIABLE);
          ++j2;
        } else {
          tmp_var_type_order.push_back(variable::VECTOR_VARIABLE);
        }
      }
      tmp_var_type_order.push_back(variable::FINITE_VARIABLE);
      finite_seq = tmp_finite_seq;
      size_t nfinite(0);
      foreach(finite_variable* v, finite_seq)
        finite_numbering_ptr_->operator[](v) = nfinite++;
      var_type_order = tmp_var_type_order;
      // Reorder m_orig_vars, m_orig_vars_indices, m_multipliers so that they
      //  follow the order of the original variables in the original dataset.
      // m_orig_vars_indices-->index in m_orig_vars
      std::map<size_t, size_t> tmp_orderstats;
      m_orig_vars.clear();
      std::vector<size_t> tmp_m_orig_vars_indices;
      m_multipliers.clear();
      for (size_t j(0); j < m_orig_vars_sorted.size(); ++j)
        tmp_orderstats[m_orig_vars_indices[j]] = j;
      for (size_t j(0); j < ds.num_finite(); ++j)
        if (tmp_orig_vars_indices_set.count(j)) { // if in merged vars
          m_orig_vars.push_back(m_orig_vars_sorted[tmp_orderstats[j]]);
          tmp_m_orig_vars_indices.push_back(j);
          m_multipliers.push_back(m_multipliers_sorted[tmp_orderstats[j]]);
        }
      m_orig_vars_indices = tmp_m_orig_vars_indices;
    } // set_merged_variables()

  void dataset_view::restrict_to_assignment(const finite_assignment& fa) {
    std::vector<size_t> offsets(fa.size(), 0); // indices of vars in records
    std::vector<size_t> vals(fa.size(), 0);    // corresponding values
    size_t i(0);
    for (finite_assignment::const_iterator it(fa.begin());
         it != fa.end(); ++it) {
      offsets[i] = this->variable_index(it->first);
      vals[i] = it->second;
      ++i;
    }
    std::vector<size_t> indices;
    i = 0;
    foreach(const record& r, this->records()) {
      bool fits(true);
      for (size_t j(0); j < offsets.size(); ++j) {
        if (r.finite(offsets[j]) != vals[j]) {
          fits = false;
          break;
        }
      }
      if (fits)
        indices.push_back(i);
      ++i;
    }
    set_record_indices(indices);
  } // restrict_to_assignment()

    // Mutating operations: datasets
    //==========================================================================

    void dataset_view::randomize(double random_seed) {
      boost::mt11213b rng(static_cast<unsigned>(random_seed));
      if (record_view == RECORD_RANGE) {
        record_indices.clear();
        for (size_t i = record_min; i < record_max; ++i)
          record_indices.push_back(i);
      } else if (record_view == RECORD_INDICES) {
      } else if (record_view == RECORD_ALL) {
        record_indices.clear();
        for (size_t i = 0; i < nrecords; ++i)
          record_indices.push_back(i);
      } else
        assert(false);
      for (size_t i = 0; i < nrecords-1; ++i) {
        size_t j((size_t)(boost::uniform_int<int>(i,nrecords-1)(rng)));
        size_t tmp(record_indices[i]);
        record_indices[i] = record_indices[j];
        record_indices[j] = tmp;
        if (weighted) {
          double tmpw(weights_[i]);
          weights_[i] = weights_[j];
          weights_[j] = tmpw;
        }
      }
      record_view = RECORD_INDICES;
    }

    // Save and load methods
    //==========================================================================

    void dataset_view::save(std::ofstream& out) const {
      out << record_view << " ";
      if (record_view == RECORD_RANGE)
        out << record_min << " " << record_max << " ";
      else if (record_view == RECORD_INDICES)
        out << record_indices << " ";
      out << (binarized_var == NULL ? "0 " : "1 ");
      if (binarized_var != NULL)
        out << binarized_var_index << " " << binary_coloring << " ";
      out << (m_new_var == NULL ? "0 " : "1 ");
      if (m_new_var != NULL)
        out << m_orig_vars_indices << " "; // TODO: FIX THIS!
      out  << "\n";
      if (vv_view != VAR_ALL)
        assert(false);
    }

    void dataset_view::save(const std::string& filename) const {
      std::ofstream out(filename.c_str(), std::ios::out);
      save(out);
      out.flush();
      out.close();
    }

    bool dataset_view::load(std::ifstream& in, finite_variable* binary_var_,
              finite_variable* m_new_var_) {
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      size_t tmpsize;
      // Record views
      if (!(is >> tmpsize))
        assert(false);
      record_view = static_cast<record_view_type>(tmpsize);
      if (record_view == RECORD_RANGE) {
        if (!(is >> record_min))
          assert(false);
        if (!(is >> record_max))
          assert(false);
      } else if (record_view == RECORD_INDICES)
        read_vec(is, record_indices);
      // Variable views
      vv_view = VAR_ALL; // TODO
      // Binarized views
      if (!(is >> tmpsize))
        assert(false);
      if (tmpsize == 1) {
        if (!(is >> binarized_var_index))
          assert(false);
        read_vec(is, binary_coloring);
        assert(binarized_var_index < ds.num_finite());
        set_binary_coloring(ds.finite_list()[binarized_var_index],
                            binary_var_, binary_coloring);
      }
      // Merged finite variable views  // TODO: FIX THIS!
      if (!(is >> tmpsize))
        assert(false);
      if (tmpsize == 1) {
        read_vec(is, m_orig_vars_indices);
        finite_var_vector orig_vars;
        foreach(size_t j, m_orig_vars_indices) {
          assert(j < ds.num_finite());
          orig_vars.push_back(ds.finite_list()[j]);
        }
        set_merged_variables(orig_vars, m_new_var_);
      }
      return true;
    }

    bool dataset_view::load(const std::string& filename, finite_variable* binary_var_,
              finite_variable* m_new_var_) {
      std::ifstream in(filename.c_str(), std::ios::in);
      bool val = load(in, binary_var_, m_new_var_);
      in.close();
      return val;
    }

} // namespace sill

#include <sill/macros_undef.hpp>

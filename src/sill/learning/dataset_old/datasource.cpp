#include <sill/learning/dataset_old/datasource.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Constructors
  //==========================================================================

  datasource::datasource() {
    datasource_info_type ds_info;
    reset(ds_info);
  }

  datasource::
  datasource(const finite_var_vector& finite_vars,
             const vector_var_vector& vector_vars,
             const std::vector<variable::variable_typenames>& var_type_order) {
    datasource_info_type ds_info(finite_vars, vector_vars, var_type_order);
    reset(ds_info);
  }

  datasource::
  datasource(const forward_range<finite_variable*>& finite_vars,
             const forward_range<vector_variable*>& vector_vars,
             const std::vector<variable::variable_typenames>& var_type_order) {
    datasource_info_type ds_info(finite_vars, vector_vars, var_type_order);
    reset(ds_info);
  }

  datasource::datasource(const datasource_info_type& info) {
    reset(info);
  }

  void datasource::reset(const datasource_info_type& info) {
//    finite_vars.clear();
//    finite_vars.insert(info.finite_seq.begin(), info.finite_seq.end());
    finite_seq = info.finite_seq;
    finite_numbering_ptr_.reset(new std::map<finite_variable*, size_t>());
    finite_class_vars = info.finite_class_vars;

//    vector_vars.clear();
//    vector_vars.insert(info.vector_seq.begin(), info.vector_seq.end());
    vector_seq = info.vector_seq;
    vector_numbering_ptr_.reset(new std::map<vector_variable*, size_t>());
    vector_class_vars = info.vector_class_vars;

    var_type_order = info.var_type_order;

    initialize();
  }

  void datasource::save(oarchive& a) const {
    a << datasource_info();
  }

  void datasource::load(iarchive& a) {
    datasource_info_type ds_info;
    a >> ds_info;
    reset(ds_info);
  }

  // Variables
  //==========================================================================

  domain datasource::variables() const {
    domain d1(finite_variables().first, finite_variables().second);
    domain d2(vector_variables().first, vector_variables().second);
    return set_union(d1, d2);
  }

  std::pair<datasource::finite_var_iterator, datasource::finite_var_iterator>
  datasource::finite_variables() const {
    return std::make_pair(finite_var_iterator(finite_numbering_ptr_->begin()),
                          finite_var_iterator(finite_numbering_ptr_->end()));
  }

  std::pair<datasource::vector_var_iterator, datasource::vector_var_iterator>
  datasource::vector_variables() const {
    return std::make_pair(vector_var_iterator(vector_numbering_ptr_->begin()),
                          vector_var_iterator(vector_numbering_ptr_->end()));
  }

  /*
  const finite_domain& datasource::finite_variables() const {
    return finite_vars;
  }

  const vector_domain& datasource::vector_variables() const {
    return vector_vars;
  }
  */

  var_vector datasource::variable_list() const {
    var_vector vars(num_variables(), NULL);
    size_t f_i = 0;
    size_t v_i = 0;
    for (size_t i = 0; i < num_variables(); ++i) {
      switch (var_type_order[i]) {
      case variable::FINITE_VARIABLE:
        vars[i] = finite_seq[f_i];
        ++f_i;
        break;
      case variable::VECTOR_VARIABLE:
        vars[i] = vector_seq[v_i];
        ++v_i;
        break;
      default:
        assert(false);
      }
    }
    assert(f_i == num_finite() && v_i == num_vector());
    return vars;
  }

  const finite_var_vector& datasource::finite_list() const {
    return finite_seq;
  }

  const vector_var_vector& datasource::vector_list() const {
    return vector_seq;
  }

  template <>
  var_vector datasource::variable_sequence<variable>() const {
    return variable_list();
  }

  template <>
  finite_var_vector datasource::variable_sequence<finite_variable>() const {
    return finite_seq;
  }

  template <>
  vector_var_vector datasource::variable_sequence<vector_variable>() const {
    return vector_seq;
  }

  finite_variable* datasource::finite_class_variable() const {
    assert(finite_class_vars.size() == 1);
    return finite_class_vars[0];
  }

  vector_variable* datasource::vector_class_variable() const {
    assert(vector_class_vars.size() == 1);
    return vector_class_vars[0];
  }

  const finite_var_vector& datasource::finite_class_variables() const {
    return finite_class_vars;
  }

  const vector_var_vector& datasource::vector_class_variables() const {
    return vector_class_vars;
  }

  bool datasource::has_variable(finite_variable* v) const {
    return (finite_numbering_ptr_->count(v) > 0);
  }

  bool datasource::has_variable(vector_variable* v) const {
    return (vector_numbering_ptr_->count(v) > 0);
  }

  bool datasource::has_variable(variable* v) const {
    switch (v.type()) {
    case variable::FINITE_VARIABLE:
      return has_variable((finite_variable*)v);
    case variable::VECTOR_VARIABLE:
      return has_variable((vector_variable*)v);
    default:
      assert(false); return false;
    }
  }

  bool datasource::has_variables(const finite_domain& vars) const {
    foreach(finite_variable* v, vars) {
      if (!has_variable(v))
        return false;
    }
    return true;
  }

  bool datasource::has_variables(const vector_domain& vars) const {
    foreach(vector_variable* v, vars) {
      if (!has_variable(v))
        return false;
    }
    return true;
  }

  bool datasource::has_variables(const domain& vars) const {
    foreach(variable* v, vars) {
      if (!has_variable(v))
        return false;
    }
    return true;
  }

  // Dimensionality of variables
  //==========================================================================

  size_t datasource::num_variables() const {
    return num_finite() + num_vector();
  }

  size_t datasource::num_finite() const {
    return finite_numbering_ptr_->size();
  }

  size_t datasource::num_vector() const {
    return vector_numbering_ptr_->size();
  }

  size_t datasource::finite_dim() const {
    return dfinite;
  }

  size_t datasource::vector_dim() const {
    return dvector;
  }

  // Getters and helpers
  //==========================================================================

  const std::vector<variable::variable_typenames>&
  datasource::variable_type_order() const {
    return var_type_order;
  }

  var_vector datasource::var_order() const {
    var_vector v;
    size_t i = 0, j = 0;
    foreach(variable::variable_typenames t, var_type_order) {
      switch(t) {
      case variable::FINITE_VARIABLE:
        v.push_back(finite_seq[i]);
        ++i;
        break;
      case variable::VECTOR_VARIABLE:
        v.push_back(vector_seq[j]);
        ++j;
        break;
      default:
        assert(false);
      }
    }
    return v;
  }

  /*
  size_t datasource::var_order_index(variable* v) const {
    return safe_get(var_order_map, v);
  }
  */

  std::map<variable*, size_t> datasource::variable_order_map() const {
    std::map<variable*, size_t> v2idx;

    size_t nf = 0, nv = 0;
    foreach(variable::variable_typenames t, var_type_order) {
      switch(t) {
      case variable::FINITE_VARIABLE:
        v2idx[finite_seq[nf]] = nf + nv;
        ++nf;
        break;
      case variable::VECTOR_VARIABLE:
        v2idx[vector_seq[nv]] = nf + nv;
        ++nv;
        break;
      default:
        assert(false);
      }
    }
    if (nf != finite_seq.size() || nv != vector_seq.size()) {
      std::cerr << "datasource::variable_order_map() found that"
                << " the variable type order has " << nf
                << " finite variables and " << nv
                << " vector variables, but the finite_seq list has size "
                << finite_seq.size() << " and the vector_seq list has size "
                << vector_seq.size() << "." << std::endl;
      assert(false);
    }

    return v2idx;
  }

  /*
  size_t datasource::variable_index(finite_variable* v) const {
    return record_index(v);
  }

  size_t datasource::variable_index(vector_variable* v) const {
    return safe_get(vector_var_order_map, v);
  }
  */

  size_t datasource::record_index(finite_variable* v) const {
    return safe_get(*finite_numbering_ptr_, v);
  }

  size_t datasource::record_index(vector_variable* v) const {
    return safe_get(*vector_numbering_ptr_, v);
  }

  uvec datasource::vector_indices(const vector_domain& vars) const {
    uvec vars_inds(vector_size(vars));
    size_t k(0);
    foreach(vector_variable* v, vars) {
      for (size_t j2(0); j2 < v.size(); ++j2) {
        vars_inds[k] = record_index(v) + j2;
        ++k;
      }
    }
    return vars_inds;
  }

  uvec datasource::vector_indices(const vector_var_vector& vars) const {
    uvec vars_inds(vector_size(vars));
    size_t k(0);
    foreach(vector_variable* v, vars) {
      for (size_t j2(0); j2 < v.size(); ++j2) {
        vars_inds[k] = record_index(v) + j2;
        ++k;
      }
    }
    return vars_inds;
  }

  const std::map<finite_variable*, size_t>&
  datasource::finite_numbering() const {
    return *finite_numbering_ptr_;
  }

  copy_ptr<std::map<finite_variable*, size_t> >
  datasource::finite_numbering_ptr() const {
    return finite_numbering_ptr_;
  }

  const std::map<vector_variable*, size_t>&
  datasource::vector_numbering() const {
    return *vector_numbering_ptr_;
  }

  copy_ptr<std::map<vector_variable*, size_t> >
  datasource::vector_numbering_ptr() const {
    return vector_numbering_ptr_;
  }

  // General info and helpers
  //==========================================================================

  datasource_info_type datasource::datasource_info() const {
    return datasource_info_type(finite_seq, vector_seq, var_type_order,
                                finite_class_vars, vector_class_vars);
  }

  bool datasource::comparable(const datasource& ds) const {
    if (//finite_vars == ds.finite_vars &&
        finite_seq == ds.finite_seq &&
        *finite_numbering_ptr_ == *(ds.finite_numbering_ptr_) &&
        dfinite == ds.dfinite &&
        finite_class_vars == ds.finite_class_vars &&
//        vector_vars == ds.vector_vars &&
        vector_seq == ds.vector_seq &&
        *vector_numbering_ptr_ == *(ds.vector_numbering_ptr_) &&
        dvector == ds.dvector &&
        vector_class_vars == ds.vector_class_vars &&
        var_type_order == ds.var_type_order)
      return true;
    else
      return false;
  }

  void datasource::print_datasource_info() const {
    std::cerr << "Datasource info:\n"
              << " " << var_type_order.size() << " variables:\n"
              << "  " << finite_seq.size() << " finite vars of arities:";
    foreach(finite_variable* v, finite_seq)
      std::cerr << " " << v.size();
    std::cerr << "\n  " << vector_seq.size() << " vector vars of sizes:";
    foreach(vector_variable* v, vector_seq)
      std::cerr << " " << v.size();
    std::cerr << std::endl;
  }

  // Setters
  //==========================================================================

  void datasource::set_finite_class_variable(finite_variable* class_var) {
    finite_class_vars.clear();
    if (class_var != NULL) {
      assert(has_variable(class_var));
      finite_class_vars.push_back(class_var);
    }
  }

  void datasource::
  set_finite_class_variables(const finite_var_vector& class_vars) {
    this->finite_class_vars.clear();
    foreach(finite_variable* v, class_vars) {
      assert(v != NULL);
      assert(has_variable(v));
      this->finite_class_vars.push_back(v);
    }
  }

  void datasource::set_finite_class_variables(const finite_domain& class_vars) {
    this->finite_class_vars.clear();
    foreach(finite_variable* v, class_vars) {
      assert(v != NULL);
      assert(has_variable(v));
      this->finite_class_vars.push_back(v);
    }
  }

  void datasource::set_vector_class_variable(vector_variable* class_var) {
    vector_class_vars.clear();
    if (class_var != NULL) {
      assert(has_variable(class_var));
      vector_class_vars.push_back(class_var);
    }
  }

  void datasource::
  set_vector_class_variables(const vector_var_vector& class_vars) {
    this->vector_class_vars.clear();
    foreach(vector_variable* v, class_vars) {
      assert(v != NULL);
      assert(has_variable(v));
      this->vector_class_vars.push_back(v);
    }
  }

  void datasource::set_vector_class_variables(const vector_domain& class_vars) {
    this->vector_class_vars.clear();
    foreach(vector_variable* v, class_vars) {
      assert(v != NULL);
      assert(has_variable(v));
      this->vector_class_vars.push_back(v);
    }
  }

  // Protected helper functions
  //==========================================================================

  void datasource::initialize() {
    size_t nfinite = 0;
    dfinite = 0;
    dvector = 0;
    finite_numbering_ptr_->clear();
    vector_numbering_ptr_->clear();

    // Compute the index maps
    foreach(finite_variable* v, finite_seq) {
      finite_numbering_ptr_->operator[](v) = nfinite++;
      dfinite += v.size();
    }
    // Compute the index maps
    foreach(vector_variable* v, vector_seq) {
      vector_numbering_ptr_->operator[](v) = dvector;
      dvector += v.size();
    }

    // Set var_type_order if necessary.
    if (var_type_order.size() == 0) {
      for (size_t j = 0; j < finite_seq.size(); ++j)
        var_type_order.push_back(variable::FINITE_VARIABLE);
      for (size_t j = 0; j < vector_seq.size(); ++j)
        var_type_order.push_back(variable::VECTOR_VARIABLE);
    }
    /*
    // Make sure the natural ordering is valid, and build var_order_map.
    var_order_map.clear();
    if (var_type_order.size() > 0) {
      size_t nf = 0, nv = 0;
      foreach(variable::variable_typenames t, var_type_order) {
        switch(t) {
        case variable::FINITE_VARIABLE:
          var_order_map[finite_seq[nf]] = nf + nv;
          ++nf;
          break;
        case variable::VECTOR_VARIABLE:
          var_order_map[vector_seq[nv]] = nf + nv;
          ++nv;
          break;
        default:
          assert(false);
        }
      }
      if (nf != finite_seq.size() || nv != vector_seq.size()) {
        std::cerr << "In datasource::initialize(): The variable type order "
                  << "has " << nf << " finite variables and " << nv
                  << " vector variables, but the finite_seq list has size "
                  << finite_seq.size() << " and the vector_seq list has size "
                  << vector_seq.size() << "." << std::endl;
        assert(false);
      }
    } else {
      for (size_t j = 0; j < finite_seq.size(); ++j) {
        var_type_order.push_back(variable::FINITE_VARIABLE);
        var_order_map[finite_seq[j]] = j;
      }
      for (size_t j = 0; j < vector_seq.size(); ++j) {
        var_type_order.push_back(variable::VECTOR_VARIABLE);
        var_order_map[vector_seq[j]] = finite_seq.size() + j;
      }
    }
    */
    /*
    // Build vector_var_order_map
    for (size_t j(0); j < vector_seq.size(); ++j)
      vector_var_order_map[vector_seq[j]] = j;
    */

  } // initialize()

  void datasource::
  convert_finite_record_old2assignment(const std::vector<size_t>& findata,
                                   finite_assignment& fa) const {
    assert(findata.size() == num_finite());
    fa.clear();
    foreach(const finite_var_index_pair& p, *finite_numbering_ptr_)
      fa[p.first] = findata[p.second];
  }

  void
  datasource::convert_vector_record_old2assignment(const vec& vecdata,
                                               vector_assignment& va) const {
    assert(vecdata.size() == dvector);
    va.clear();
    foreach(const vector_var_index_pair& p, *vector_numbering_ptr_) {
      vec tmpvec(p.first->size());
      for(size_t j = 0; j < p.first->size(); j++)
        tmpvec[j] = vecdata[j+p.second];
      va[p.first] = tmpvec;
    }
  }

  void datasource::
  convert_finite_assignment2record(const finite_assignment& fa,
                                   std::vector<size_t>& findata) const {
    finite_assignment2vector(fa, finite_seq, findata);
  }

  void datasource::add_finite_variable(finite_variable* v, bool make_class) {
    assert(!has_variable(v));
//    finite_vars.insert(v);
    finite_seq.push_back(v);
    finite_numbering_ptr_->operator[](v) = finite_seq.size() - 1;
    dfinite += v.size();
    if (make_class)
      finite_class_vars.push_back(v);
    var_type_order.push_back(variable::FINITE_VARIABLE);
  }

  void datasource::add_vector_variable(vector_variable* v, bool make_class) {
    assert(!has_variable(v));
//    vector_vars.insert(v);
    vector_seq.push_back(v);
    vector_numbering_ptr_->operator[](v) = vector_seq.size() - 1;
    dvector += v.size();
    if (make_class)
      vector_class_vars.push_back(v);
    var_type_order.push_back(variable::VECTOR_VARIABLE);
  }

} // namespace sill

#include <sill/macros_undef.hpp>

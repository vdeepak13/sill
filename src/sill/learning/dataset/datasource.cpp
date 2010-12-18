
#include <sill/learning/dataset/datasource.hpp>
#include <sill/learning/dataset/record_conversions.hpp>

#include <sill/macros_def.hpp>

namespace sill {

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
      dfinite += v->size();
    }
    // Compute the index maps
    foreach(vector_variable* v, vector_seq) {
      vector_numbering_ptr_->operator[](v) = dvector;
      dvector += v->size();
    }

    // Make sure the natural ordering is valid, and built var_order_map.
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
    // Build vector_var_order_map
    for (size_t j(0); j < vector_seq.size(); ++j)
      vector_var_order_map[vector_seq[j]] = j;

  } // initialize()

  void datasource::
  convert_finite_record2assignment(const std::vector<size_t>& findata,
                                   finite_assignment& fa) const {
    assert(findata.size() == finite_vars.size());
    fa.clear();
    foreach(const finite_var_index_pair& p, *finite_numbering_ptr_)
      fa[p.first] = findata[p.second];
  }

  void datasource::
  convert_vector_record2assignment(const vec& vecdata,
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
    finite_assignment2record(fa, findata, finite_seq);
  }

  void datasource::
  convert_vector_assignment2record(const vector_assignment& va,
                                   vec& vecdata) const {
    vector_assignment2record(va, vecdata, vector_seq);
  }

  void datasource::add_finite_variable(finite_variable* v, bool make_class) {
    if (finite_vars.count(v))
      assert(false);
    finite_vars.insert(v);
    finite_seq.push_back(v);
    finite_numbering_ptr_->operator[](v) = finite_seq.size() - 1;
    dfinite += v->size();
    if (make_class)
      finite_class_vars.push_back(v);
    var_type_order.push_back(variable::FINITE_VARIABLE);
  }

  void datasource::add_vector_variable(vector_variable* v, bool make_class) {
    if (vector_vars.count(v))
      assert(false);
    vector_vars.insert(v);
    vector_seq.push_back(v);
    vector_numbering_ptr_->operator[](v) = vector_seq.size() - 1;
    dvector += v->size();
    if (make_class)
      vector_class_vars.push_back(v);
    var_type_order.push_back(variable::VECTOR_VARIABLE);
  }


  // Getters and helpers
  //==========================================================================

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

  size_t datasource::var_order_index(variable* v) const {
    return safe_get(var_order_map, v);
  }

  ivec datasource::vector_indices(const vector_domain& vars) const {
    ivec vars_inds(vector_size(vars));
    size_t k(0);
    foreach(vector_variable* v, vars) {
      for (size_t j2(0); j2 < v->size(); ++j2) {
        vars_inds[k] = record_index(v) + j2;
        ++k;
      }
    }
    return vars_inds;
  }

  ivec datasource::vector_indices(const vector_var_vector& vars) const {
    ivec vars_inds(vector_size(vars));
    size_t k(0);
    foreach(vector_variable* v, vars) {
      for (size_t j2(0); j2 < v->size(); ++j2) {
        vars_inds[k] = record_index(v) + j2;
        ++k;
      }
    }
    return vars_inds;
  }

  bool datasource::comparable(const datasource& ds) const {
    if (finite_vars == ds.finite_vars && finite_seq == ds.finite_seq &&
        *finite_numbering_ptr_ == *(ds.finite_numbering_ptr_) &&
        dfinite == ds.dfinite &&
        finite_class_vars == ds.finite_class_vars &&
        vector_vars == ds.vector_vars && vector_seq == ds.vector_seq &&
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
      std::cerr << " " << v->size();
    std::cerr << "\n  " << vector_seq.size() << " vector vars of sizes:";
    foreach(vector_variable* v, vector_seq)
      std::cerr << " " << v->size();
    std::cerr << std::endl;
  }

  // Setters
  //==========================================================================

  void datasource::set_finite_class_variable(finite_variable* class_var) {
    finite_class_vars.clear();
    if (class_var != NULL) {
      assert(finite_vars.count(class_var));
      finite_class_vars.push_back(class_var);
    }
  }

  void datasource::
  set_finite_class_variables(const finite_var_vector& class_vars) {
    this->finite_class_vars.clear();
    foreach(finite_variable* v, class_vars) {
      assert(v != NULL);
      assert(finite_vars.count(v));
      this->finite_class_vars.push_back(v);
    }
  }

  void datasource::set_finite_class_variables(const finite_domain& class_vars) {
    this->finite_class_vars.clear();
    foreach(finite_variable* v, class_vars) {
      assert(v != NULL);
      assert(finite_vars.count(v));
      this->finite_class_vars.push_back(v);
    }
  }

  void datasource::set_vector_class_variable(vector_variable* class_var) {
    vector_class_vars.clear();
    if (class_var != NULL) {
      assert(vector_vars.count(class_var));
      vector_class_vars.push_back(class_var);
    }
  }

  void datasource::
  set_vector_class_variables(const vector_var_vector& class_vars) {
    this->vector_class_vars.clear();
    foreach(vector_variable* v, class_vars) {
      assert(v != NULL);
      assert(vector_vars.count(v));
      this->vector_class_vars.push_back(v);
    }
  }

  void datasource::set_vector_class_variables(const vector_domain& class_vars) {
    this->vector_class_vars.clear();
    foreach(vector_variable* v, class_vars) {
      assert(v != NULL);
      assert(vector_vars.count(v));
      this->vector_class_vars.push_back(v);
    }
  }

} // namespace sill

#include <sill/macros_undef.hpp>

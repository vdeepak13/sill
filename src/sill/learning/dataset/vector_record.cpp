#include <sill/learning/dataset/vector_record.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Getters and helpers
  //==========================================================================

  bool vector_record::has_variable(vector_variable* v) const {
    return (vector_numbering_ptr->count(v) != 0);
  }

  sill::vector_assignment vector_record::vector_assignment() const {
    sill::vector_assignment a;
    foreach(const vector_var_index_pair& p, *vector_numbering_ptr) {
      vec v(p.first->size());
      for(size_t j = 0; j < p.first->size(); ++j)
        v[j] = vec_ptr->operator[](j+p.second);
      a[p.first] = v;
    }
    return a;
  }

  sill::vector_assignment
  vector_record::assignment(const vector_domain& X) const {
    sill::vector_assignment a;
    foreach(vector_variable* v, X) {
      size_t v_index(safe_get(*vector_numbering_ptr, v));
      vec val(v->size());
      for(size_t j(0); j < v->size(); ++j)
        val[j] = vec_ptr->operator[](v_index + j);
      a[v] = val;
    }
    return a;
  }

  void
  vector_record::
  add_assignment(const vector_domain& X, sill::vector_assignment& a) const {
    foreach(vector_variable* v, X) {
      size_t v_index(safe_get(*vector_numbering_ptr, v));
      vec val(v->size());
      for(size_t j(0); j < v->size(); ++j)
        val[j] = vec_ptr->operator[](v_index + j);
      a[v] = val;
    }
  }

  vector_var_vector vector_record::vector_list() const {
    vector_var_vector vlist(vector_numbering_ptr->size(), NULL);
    for (std::map<vector_variable*,size_t>::const_iterator
           it(vector_numbering_ptr->begin());
         it != vector_numbering_ptr->end(); ++it)
      vlist[it->second] = it->first;
    return vlist;
  }

  vector_domain vector_record::variables() const {
    return keys(*vector_numbering_ptr);
  }

    vector_record& vector_record::operator=(const vector_record& rec) {
      vector_numbering_ptr = rec.vector_numbering_ptr;
      if (vec_own) {
        vec_ptr->resize(rec.vec_ptr->size());
        for (size_t j(0); j < rec.vec_ptr->size(); ++j)
          vec_ptr->operator[](j) = rec.vec_ptr->operator[](j);
      } else {
        vec_own = true;
        vec_ptr = new vec(*(rec.vec_ptr));
      }
      return *this;
    }

  vector_record& vector_record::operator=(const sill::vector_assignment& a) {
    size_t a_vars_size(0);
    foreach(const sill::vector_assignment::value_type& a_val, a) {
      a_vars_size += a_val.first->size();
    }
    if (!vec_own) {
      vec_ptr = new vec(a_vars_size, 0.);
      vec_own = true;
    } else {
      vec_ptr->resize(a_vars_size);
    }
    foreach(const vector_var_index_pair& p, *vector_numbering_ptr) {
      const vec& v = safe_get(a, p.first);
      for (size_t j = 0; j < p.first->size(); ++j)
        vec_ptr->operator[](p.second + j) = v[j];
    }
    return *this;
  }

  // Mutating operations
  //==========================================================================

  void vector_record::clear() {
    vector_numbering_ptr->clear();
    if (vec_own) {
      vec_ptr->clear();
    } else {
      vec_own = true;
      vec_ptr = new vec();
    }
  }

    void
    vector_record::reset
    (copy_ptr<std::map<vector_variable*, size_t> > vector_numbering_ptr,
     size_t vector_dim) {
      this->vector_numbering_ptr = vector_numbering_ptr;
      if (vec_own) {
        vec_ptr->resize(vector_dim);
      } else {
        vec_own = true;
        vec_ptr = new vec(vector_dim);
      }
    }

} // namespace sill

#include <sill/macros_undef.hpp>

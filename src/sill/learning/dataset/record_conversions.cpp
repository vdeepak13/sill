
#include <sill/learning/dataset/record_conversions.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  void
  vector_record2vector(const vector_record& r, const vector_var_vector& vars,
                       vec& vals) {
    vals.resize(vector_size(vars));
    size_t i(0); // index into vals
    foreach(vector_variable* v, vars) {
      for (size_t j(0); j < v->size(); ++j) {
        vals[i] = r.vector(v, j);
        ++i;
      }
    }
  }

  void
  finite_assignment2vector(const finite_assignment& fa,
                           const finite_var_vector& finite_seq,
                           std::vector<size_t>& findata) {
    if (findata.size() != finite_seq.size())
      findata.resize(finite_seq.size());
    finite_assignment::const_iterator fa_end(fa.end());
    for (size_t i(0); i < finite_seq.size(); i++) {
      finite_assignment::const_iterator it(fa.find(finite_seq[i]));
      assert(it != fa_end);
      findata[i] = it->second;
    }
  }

  void
  vector_assignment2vector(const vector_assignment& va,
                           const vector_var_vector& vector_seq,
                           vec& vecdata) {
    vecdata.resize(vector_size(vector_seq));
    size_t k(0); // index into vecdata
    vector_assignment::const_iterator va_end = va.end();
    foreach(vector_variable* v, vector_seq) {
      vector_assignment::const_iterator it(va.find(v));
      if (it == va_end) {
        throw std::runtime_error("vector_assignment2vector given vector_seq with variables not appearing in given assignment.");
      }
      const vec& tmpvec = it->second;
      for (size_t j(0); j < v->size(); j++)
        vecdata[k + j] = tmpvec[j];
      k += v->size();
    }
  }

  void fill_record_with_assignment(record& r, const assignment& a,
                                   const var_map& vmap) {
    for (std::map<finite_variable*, size_t>::const_iterator it =
           r.finite_numbering_ptr->begin();
         it != r.finite_numbering_ptr->end();
         ++it) {
      r.finite(it->second) =
        safe_get(a.finite(),
                 (finite_variable*)(safe_get(vmap,(variable*)(it->first))));
    }
    for (std::map<vector_variable*, size_t>::const_iterator it =
           r.vector_numbering_ptr->begin();
         it != r.vector_numbering_ptr->end();
         ++it) {
      r.vector().set_subvector
        (irange(it->second, it->second + it->first->size()),
         safe_get(a.vector(),
                  (vector_variable*)(safe_get(vmap, (variable*)(it->first)))));
    }
  }

  void fill_record_with_assignment(finite_record& r, const finite_assignment& a,
                                   const finite_var_map& vmap) {
    for (std::map<finite_variable*, size_t>::const_iterator it =
           r.finite_numbering_ptr->begin();
         it != r.finite_numbering_ptr->end();
         ++it) {
      r.finite(it->second) = safe_get(a, safe_get(vmap, it->first));
    }
  }

  void fill_record_with_assignment(vector_record& r, const vector_assignment& a,
                                   const vector_var_map& vmap) {
    for (std::map<vector_variable*, size_t>::const_iterator it =
           r.vector_numbering_ptr->begin();
         it != r.vector_numbering_ptr->end();
         ++it) {
      r.vector().set_subvector
        (irange(it->second, it->second + it->first->size()),
         safe_get(a, safe_get(vmap, it->first)));
    }
  }

  void fill_record_with_record(record& to, const record& from,
                               const var_map& vmap) {
    for (std::map<finite_variable*, size_t>::const_iterator it =
           to.finite_numbering_ptr->begin();
         it != to.finite_numbering_ptr->end();
         ++it) {
      to.finite(it->second) =
        from.finite((finite_variable*)(safe_get(vmap,(variable*)(it->first))));
    }
    for (std::map<vector_variable*, size_t>::const_iterator it =
           to.vector_numbering_ptr->begin();
         it != to.vector_numbering_ptr->end();
         ++it) {
      size_t i =
        safe_get(*(from.vector_numbering_ptr),
                 (vector_variable*)(safe_get(vmap,(variable*)(it->first))));
      for (size_t j(it->second); j < it->second + it->first->size(); ++j) {
        to.vector(j) = from.vector(i);
        ++i;
      }
    }
  }

  void fill_record_with_record(finite_record& to, const finite_record& from,
                               const finite_var_map& vmap) {
    for (std::map<finite_variable*, size_t>::const_iterator it =
           to.finite_numbering_ptr->begin();
         it != to.finite_numbering_ptr->end();
         ++it) {
      to.finite(it->second) = from.finite(safe_get(vmap,it->first));
    }
  }

  void fill_record_with_record(vector_record& to, const vector_record& from,
                               const vector_var_map& vmap) {
    for (std::map<vector_variable*, size_t>::const_iterator it =
           to.vector_numbering_ptr->begin();
         it != to.vector_numbering_ptr->end();
         ++it) {
      size_t i(safe_get(*(from.vector_numbering_ptr),
                        safe_get(vmap,it->first)));
      for (size_t j(it->second); j < it->second + it->first->size(); ++j) {
        to.vector(j) = from.vector(i);
        ++i;
      }
    }
  }

} // namespace sill

#include <sill/macros_undef.hpp>

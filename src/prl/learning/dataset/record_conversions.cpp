
#include <sill/learning/dataset/record_conversions.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  void
  finite_assignment2record(const finite_assignment& fa,
                           std::vector<size_t>& findata,
                           const finite_var_vector& finite_seq) {
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
  vector_assignment2record(const vector_assignment& va,
                           vec& vecdata,
                           const vector_var_vector& vector_seq) {
    size_t k(0); // index into vecdata
    vector_assignment::const_iterator va_end = va.end();
    for (size_t i(0); i < vector_seq.size(); i++) {
      vector_assignment::const_iterator it(va.find(vector_seq[i]));
      assert(it != va_end);
      const vec& tmpvec = it->second;
      if (k + vector_seq[i]->size() > vecdata.size())
        vecdata.resize(k + vector_seq[i]->size(), true);
      for (size_t j(0); j < vector_seq[i]->size(); j++)
        vecdata[k + j] = tmpvec[j];
      k += vector_seq[i]->size();
    }
    if (vecdata.size() > k)
      vecdata.resize(k, true);
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

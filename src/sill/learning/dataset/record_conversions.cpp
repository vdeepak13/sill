
#include <sill/learning/dataset/record_conversions.hpp>

#include <sill/macros_def.hpp>

namespace sill {

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

  /*
  void fill_record_with_assignment(finite_record& r, const finite_assignment& a,
                                   const finite_var_map& vmap) {
    for (std::map<finite_variable*, size_t>::const_iterator it =
           r.finite_numbering_ptr->begin();
         it != r.finite_numbering_ptr->end();
         ++it) {
      r.finite(it->second) = safe_get(a, safe_get(vmap, it->first));
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
  */

} // namespace sill

#include <sill/macros_undef.hpp>

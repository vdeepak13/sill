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

  void
  vector_assignment2vector(const vector_assignment& va,
                           const vector_var_vector& vector_seq,
                           vec& vecdata) {
    vecdata.set_size(vector_size(vector_seq));
    size_t k(0); // index into vecdata
    vector_assignment::const_iterator va_end = va.end();
    foreach(vector_variable* v, vector_seq) {
      vector_assignment::const_iterator it(va.find(v));
      if (it == va_end) {
        throw std::runtime_error
          (std::string("vector_assignment2vector(va,vector_seq,vecdata)") +
           " given vector_seq with variables not appearing in given" +
           " assignment.");
      }
      const vec& tmpvec = it->second;
      for (size_t j(0); j < v->size(); j++)
        vecdata[k + j] = tmpvec[j];
      k += v->size();
    }
  }

} // namespace sill

#include <sill/macros_undef.hpp>

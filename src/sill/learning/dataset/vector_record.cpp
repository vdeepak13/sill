#include <sill/learning/dataset/datasource.hpp>
#include <sill/learning/dataset/vector_record.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Getters and helpers
  //==========================================================================


  sill::vector_assignment
  vector_record::assignment(const vector_domain& X) const {
    sill::vector_assignment a;
    foreach(vector_variable* v, X) {
      size_t v_index(safe_get(*vector_numbering_ptr, v));
      vector_type val(v->size());
      for(size_t j(0); j < v->size(); ++j)
        val[j] = vec_ptr->operator[](v_index + j);
      a[v] = val;
    }
    return a;
  }

  void
  vector_record::
  add_assignment(const vector_domain& X, sill::vector_assignment& a) const {

  vector_var_vector vector_record::vector_list() const {

  vector_domain vector_record::variables() const {

  vector_record& vector_record::operator=(const vector_record& rec) {

  vector_record& vector_record::operator=(const sill::vector_assignment& a) {

  // Mutating operations
  //==========================================================================

  void vector_record::clear() {

  void
  vector_record::reset
  (copy_ptr<std::map<vector_variable*, size_t> > vector_numbering_ptr,
   size_t vector_dim) {

  void
  vector_record::reset(const datasource_info_type& ds_info) {

} // namespace sill

#include <sill/macros_undef.hpp>

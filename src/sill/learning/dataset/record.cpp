#include <sill/base/stl_util.hpp>
#include <sill/learning/dataset/record.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Constructors
  //==========================================================================

  record::record(const datasource_info_type& ds_info)
    : finite_record(ds_info.finite_seq),
      vector_record(ds_info.vector_seq, vector_size(ds_info.vector_seq)) {
  }

  bool record::operator==(const record& other) const {
    if (finite_record::operator==(other) &&
        vector_record::operator==(other))
      return true;
    else
      return false;
  }

  bool record::operator!=(const record& other) const {
    return !operator==(other);
  }

  // Getters and helpers
  //==========================================================================

  sill::assignment record::assignment(const domain& X) const {
    sill::assignment a;
    foreach(variable* v, X) {
      switch(v->get_variable_type()) {
      case variable::FINITE_VARIABLE:
        {
          finite_variable* vf = dynamic_cast<finite_variable*>(v);
          a.finite()[vf] = this->finite(vf);
        }
        break;
      case variable::VECTOR_VARIABLE:
        {
          vector_variable* vv = dynamic_cast<vector_variable*>(v);
          size_t v_index(safe_get(*vector_numbering_ptr, vv));
          vec val(vv->size());
          for(size_t j(0); j < vv->size(); ++j)
            val[j] = vec_ptr->operator[](v_index + j);
          a.vector()[vv] = val;
        }
        break;
      default:
        assert(false);
      }
    }
    return a;
  }

  sill::finite_assignment record::assignment(const finite_domain& X) const {
    return finite_record::assignment(X);
  }

  sill::vector_assignment record::assignment(const vector_domain& X) const {
    return vector_record::assignment(X);
  }

  void
  record::add_assignment(const domain& X, sill::assignment& a) const {
    foreach(variable* v, X) {
      switch(v->get_variable_type()) {
      case variable::FINITE_VARIABLE:
        {
          finite_variable* vf = dynamic_cast<finite_variable*>(v);
          a.finite()[vf] = this->finite(vf);
        }
        break;
      case variable::VECTOR_VARIABLE:
        {
          vector_variable* vv = dynamic_cast<vector_variable*>(v);
          size_t v_index(safe_get(*vector_numbering_ptr, vv));
          vec val(vv->size());
          for(size_t j(0); j < vv->size(); ++j)
            val[j] = vec_ptr->operator[](v_index + j);
          a.vector()[vv] = val;
        }
        break;
      default:
        assert(false);
      }
    }
  }

  void
  record::
  add_assignment(const finite_domain& X, sill::finite_assignment& a) const {
    finite_record::add_assignment(X, a);
  }

  void
  record::
  add_assignment(const vector_domain& X, sill::vector_assignment& a) const {
    vector_record::add_assignment(X, a);
  }

} // namespace sill

#include <sill/macros_undef.hpp>

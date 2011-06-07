
#include <sill/learning/dataset/finite_record_iterator.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Private data and methods
  //==========================================================================
  void finite_record_iterator::set_val() const {
    if (iter != r_ptr->finite_numbering_ptr->end()) {
      val.first = iter->first;
      val.second = r_ptr->finite(iter->second);
    } else {
      val = std::make_pair(static_cast<finite_variable*>(NULL), 0);
    }
  }

  // Public methods
  //==========================================================================

  finite_record_iterator::finite_record_iterator
  (const finite_record& r, finite_variable* v)
    : r_ptr(&r), iter(r.finite_numbering_ptr->find(v)), val(NULL, 0) {
  }

  finite_record_iterator& finite_record_iterator::operator++() {
    assert(iter != r_ptr->finite_numbering_ptr->end());
    ++iter;
    return *this;
  }

  finite_record_iterator finite_record_iterator::operator++(int) {
    finite_record_iterator tmp(*this);
    ++(*this);
    return tmp;
  }

  const std::pair<finite_variable*, size_t>&
  finite_record_iterator::operator*() const {
    set_val();
    return val;
  }

  const std::pair<finite_variable*, size_t>* const
  finite_record_iterator::operator->() const {
    set_val();
    return &val;
  }

  bool
  finite_record_iterator::operator==(const finite_record_iterator& it) const {
    if (r_ptr == it.r_ptr && iter == it.iter)
      return true;
    return false;
  }

  bool
  finite_record_iterator::operator!=(const finite_record_iterator& it) const {
    return !operator==(it);
  }

  bool finite_record_iterator::is_end() const {
    return (iter == r_ptr->finite_numbering_ptr->end());
  }

} // namespace sill

#include <sill/macros_undef.hpp>

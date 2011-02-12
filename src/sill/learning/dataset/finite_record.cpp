#include <sill/learning/dataset/finite_record.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Getters and helpers
  //==========================================================================

  bool finite_record::has_variable(finite_variable* v) const {
    return (finite_numbering_ptr->count(v) != 0);
  }

  sill::finite_assignment finite_record::finite_assignment() const {
    sill::finite_assignment a;
    foreach(const finite_var_index_pair& p, *finite_numbering_ptr) {
      a[p.first] = fin_ptr->operator[](p.second);
    }
    return a;
  }

  sill::finite_assignment
  finite_record::assignment(const finite_domain& X) const {
    sill::finite_assignment a;
    foreach(finite_variable* v, X) {
      size_t v_index(safe_get(*finite_numbering_ptr, v));
      a[v] = fin_ptr->operator[](v_index);
    }
    return a;
  }

  void
  finite_record::
  add_assignment(const finite_domain& X, sill::finite_assignment& a) const {
    foreach(finite_variable* v, X) {
      size_t v_index(safe_get(*finite_numbering_ptr, v));
      a[v] = fin_ptr->operator[](v_index);
    }
  }

  finite_var_vector finite_record::finite_list() const {
    finite_var_vector flist(finite_numbering_ptr->size(), NULL);
    for (std::map<finite_variable*,size_t>::const_iterator
           it(finite_numbering_ptr->begin());
         it != finite_numbering_ptr->end(); ++it)
      flist[it->second] = it->first;
    return flist;
  }

  finite_domain finite_record::variables() const {
    return keys(*finite_numbering_ptr);
  }

  finite_record_iterator finite_record::find(finite_variable* v) const {
    return finite_record_iterator(*this, v);
  }

  finite_record_iterator finite_record::end() const {
    return finite_record_iterator(*this, NULL);
  }

  finite_record& finite_record::operator=(const finite_record& rec) {
    finite_numbering_ptr = rec.finite_numbering_ptr;
    if (fin_own) {
      fin_ptr->resize(rec.fin_ptr->size());
      for (size_t j(0); j < rec.fin_ptr->size(); ++j)
        fin_ptr->operator[](j) = rec.fin_ptr->operator[](j);
    } else {
      fin_own = true;
      fin_ptr = new std::vector<size_t>(*(rec.fin_ptr));
    }
    return *this;
  }

  finite_record& finite_record::operator=(const sill::finite_assignment& a) {
    if (!fin_own) {
      fin_ptr = new std::vector<size_t>(finite_numbering_ptr->size(), 0);
      fin_own = true;
    } else {
      fin_ptr->resize(finite_numbering_ptr->size());
    }
    foreach(const finite_var_index_pair& p, *finite_numbering_ptr)
      fin_ptr->operator[](p.second) = safe_get(a, p.first);
    return *this;
  }

  // Mutating operations
  //============================================================================

  void finite_record::clear() {
    finite_numbering_ptr->clear();
    if (fin_own) {
      fin_ptr->clear();
    } else {
      fin_own = true;
      fin_ptr = new std::vector<size_t>();
    }
  }

  void
  finite_record::reset
  (copy_ptr<std::map<finite_variable*, size_t> > finite_numbering_ptr) {
    this->finite_numbering_ptr = finite_numbering_ptr;
    if (fin_own) {
      fin_ptr->resize(finite_numbering_ptr->size());
    } else {
      fin_own = true;
      fin_ptr = new std::vector<size_t>(finite_numbering_ptr->size());
    }
  }

} // namespace sill

#include <sill/macros_undef.hpp>

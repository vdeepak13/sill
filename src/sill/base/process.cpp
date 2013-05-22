#include <sill/base/process.hpp>
#include <sill/base/universe.hpp>
#include <sill/base/timed_process.hpp>
#include <iostream>

namespace sill {

  std::ostream& operator<<(std::ostream& out, process* p) {
    out << std::string(*p);
    return out;
  }

  process::process_typenames process::get_process_type() const {
    if (typeid(*this) == typeid(const timed_process<finite_variable>)) {
      return process::TIMED_PROCESS_FINITE;
    }
    else if (typeid(*this) == typeid(const timed_process<vector_variable>)) {
      return process::TIMED_PROCESS_VECTOR;
    }
    assert(false);
    return process::process_typenames(0);
  }

  oarchive& operator<<(oarchive & ar, process* v) {
    ar << v->id();
    return ar;
  }

  iarchive& operator>>(iarchive & ar, process* &p) {
    // extract the universe from the hints
    assert(ar.universe() != NULL);
    // read the variable id
    size_t id;
    ar >> id;
    //find the variable
    p = ar.universe()->proc_from_id(id);
    assert(p != NULL);
    return ar;
  }

  void dynamic_deep_serialize(oarchive &a, process* i) {
    a << (i == NULL);
    if (i == NULL) return;
    size_t ptype = size_t(i->get_process_type());
    a << ptype;
    i->save(a);
  }

  void dynamic_deep_deserialize(iarchive &a, process*& p) {
    bool isnull;
    a >> isnull;
    if (isnull) {
      p = NULL;
      return;
    }
    size_t ptype;
    a >> ptype;

    switch(ptype) {
    case process::TIMED_PROCESS_FINITE:
      p = new timed_process<finite_variable>;
      break;
    case process::TIMED_PROCESS_VECTOR:
      p = new timed_process<vector_variable>;
      break;
    default:
      assert(false);
    }
    p->load(a);
  }
}

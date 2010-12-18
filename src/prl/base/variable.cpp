#include <string>

#include <sill/base/variable.hpp>
#include <sill/base/universe.hpp>
#include <sill/base/finite_variable.hpp>
#include <sill/base/vector_variable.hpp>
#include <sill/base/process.hpp>
#include <sill/base/finite_assignment_iterator.hpp>
#include <sill/serialization/serialize.hpp>

#include <sill/macros_def.hpp>

// All variables are implemented here (in a single file) to cut down 
// on the compilation time

namespace sill {

  std::ostream& operator<<(std::ostream& out, variable* v) {
    out << std::string(*v) << (void*)(v);
    return out;
  }

  variable::variable_typenames variable::get_variable_type() const{
    if (typeid(*this) == typeid(const finite_variable)) {
      return FINITE_VARIABLE;
    }
    else if (typeid(*this) == typeid(const vector_variable)) {
      return VECTOR_VARIABLE;
    }
    assert(false);
    return variable::variable_typenames(0);  
  }

  void variable::save(oarchive& ar) const{
    ar << name_ << id_;
    ar << (process_ != NULL);
  
    if (process_ != NULL) {
      ar << process_;
    }
  }

  void variable::load(iarchive & ar) {
    ar >> name_ >> id_;
    bool hasprocess;
    ar >> hasprocess;
    if (hasprocess) {
      ar >> process_;
    }
  }

  // serialization of the pointer redirects to the respective vector and finite
  // variable pointer serializers
  oarchive& operator<<(oarchive& ar, variable* const& v) {
    ar << (v->process() != NULL);
    if (v->process() == NULL) {
      ar << v->id();
    } else {
      ar << v->process();
      v->process()->save_variable(ar, v);
    }
    return ar;
  }

  iarchive& operator>>(iarchive& ar, variable*& v) {
    bool has_process;
    ar >> has_process;
    assert(ar.universe() != NULL);
    if (has_process) {
      process* p;
      ar >> p;
      v = p->load_variable(ar);
    } else {
      // read the variable id
      size_t id;
      ar >> id;
      //find the variable
      v = ar.universe()->var_from_id(id);
    }
    return ar;
  }

  void dynamic_deep_serialize(oarchive& a, variable* const& i) {
    a << (i == NULL);
    if (i == NULL) return;

    a << size_t(i->get_variable_type());
    i->save(a);
  }

  void dynamic_deep_deserialize(iarchive& a, variable*& v) {
    bool isnull;
    a >> isnull;
  
    if (isnull) {
      v = NULL;
      return;
    }
  
    size_t vtype;
    a >> vtype;
    switch(vtype) {
    case variable::FINITE_VARIABLE:
      v = new finite_variable();
      break;
    case variable::VECTOR_VARIABLE:
      v = new vector_variable();
      break;
    default:
      assert(false);
    }
    v->load(a);
  }


} // namespace sill


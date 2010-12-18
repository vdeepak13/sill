#include <sill/base/universe.hpp>
#include <sill/base/process.hpp>
#include <sill/base/timed_process.hpp>
#include <sill/macros_def.hpp>

namespace sill {

  universe::~universe() {
    foreach(variable *v, vars_vector) {
      delete v;
    }
    foreach(process *p, procs_vector) {
      delete p;
    }
  }

  void universe::register_variable_id(variable* v) {
    v->set_id(vars_vector.size());
    vars_vector.push_back(v);
    vars[v->name()] = v;
  }
  
  void universe::register_process_id(process* p) {
    p->set_id(procs_vector.size());
    procs_vector.push_back(p);
    procs[p->name()] = p;
  }
  
  finite_variable* 
  universe::new_finite_variable(const std::string& name, size_t size) {
    // new variable
    finite_variable* v = new finite_variable(name, size);
    register_variable_id(v);
    return v;
  }

  vector_variable* 
  universe::new_vector_variable(const std::string& name, size_t size) {
    // new variable
    vector_variable* v = new vector_variable(name, size);
    register_variable_id(v);
    return v;
  }

  template <>
  finite_variable* universe::new_variable<finite_variable>(size_t size) {
    return new_finite_variable(size);
  }

  template <>
  vector_variable* universe::new_variable<vector_variable>(size_t size) {
    return new_vector_variable(size);
  }

  void universe::add_impl(variable* v) {
    assert(v->process() == NULL);
    register_variable_id(v);
  }

  void universe::add_impl(process* p) {
    register_process_id(p);
  }

  void universe::save(oarchive& a) const {
    a << procs_vector.size();
    foreach(process* p, procs_vector) {
      sill::dynamic_deep_serialize(a, p);
    }

    a <<  vars_vector.size();
    foreach(variable* v, vars_vector) {
      sill::dynamic_deep_serialize(a, v);
    }
  }
  
  void universe::load(iarchive& a) {
    size_t varsvsize, procvsize;
    procs.clear();
    vars.clear();
    // deserialize the processes first
    a >> procvsize;
    for (size_t i = 0; i < procvsize; ++i) {
      process *p;
      sill::dynamic_deep_deserialize(a, p);
      procs_vector.push_back(p);
      procs[p->name()] = p;
    }

    // deserialize the variables
    a >> varsvsize;
    for (size_t i = 0; i < varsvsize; ++i) {
      variable *v;
      sill::dynamic_deep_deserialize(a, v);
      vars_vector.push_back(v);
      vars[v->name()] = v;
    }
  }

} // namespace sill


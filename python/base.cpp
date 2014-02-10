#include <sill/base/universe.hpp>
#include <sill/base/variable.hpp>
#include <sill/base/finite_variable.hpp>
#include <sill/base/finite_assignment.hpp>
#include <sill/base/vector_variable.hpp>
#include <sill/base/vector_assignment.hpp>

#include <boost/python.hpp>

#include "converters.hpp"

namespace sill {
  std::ostream& operator<<(std::ostream& out, const sill::variable& v) {
    out << const_cast<sill::variable*>(&v);
    return out;
  }
}

void def_base() {
  using namespace boost::python;
  using namespace sill;

  using self_ns::str;
  using self_ns::repr;
  
  // todo serialization
  class_<variable, boost::noncopyable>("variable", no_init)
    .def("name", &variable::name, return_value_policy<copy_const_reference>())
    .def("size", &variable::size)
    .def("id", &variable::id)
    .def(str(self))
    .def(repr(self));
    //return_internal_reference<>())

  // domain, var_vector, var_map

  bool (finite_variable::*compat_ff)(finite_variable*) const =
    &finite_variable::type_compatible;
  bool (finite_variable::*compat_fb)(variable*)        const =
    &finite_variable::type_compatible;
  bool (vector_variable::*compat_vv)(vector_variable*) const =
    &vector_variable::type_compatible;
  bool (vector_variable::*compat_vb)(variable*)        const =
    &vector_variable::type_compatible;

  class_<finite_variable, bases<variable>, boost::noncopyable>("finite_variable", no_init)
    .def("type_compatible", compat_ff)
    .def("type_compatible", compat_fb);
    //    .def(self_ns::str(self_ns::self));
  // finite_domain, finite_var_vector, finite_var_map

  class_<vector_variable, bases<variable>, boost::noncopyable>("vector_variable", no_init)
    .def("type_compatible", compat_vv)
    .def("type_compatible", compat_vb);
    //    .def(self_ns::str(self));
  // vector_domain, vector_var_vector, vector_var_map
  
  finite_variable* (universe::*new_fv)(const std::string&, size_t) =
    &universe::new_finite_variable;
  vector_variable* (universe::*new_vv)(const std::string&, size_t) =
    &universe::new_vector_variable;
  class_<universe>("universe")
    .def("new_finite_variable", new_fv, return_internal_reference<>())
    .def("new_vector_variable", new_vv, return_internal_reference<>());

  finite_assignment (*make_fa)(finite_variable* var, size_t)     = &make_assignment;
  vector_assignment (*make_va)(vector_variable* var, const vec&) = &make_assignment;

  def("make_assignment", make_fa);
  def("make_assignment", make_va);

  //variable_converters();
  
  // finite_assignment
  //def("assignment_agreement", assignment_agreement);

  // vector_assignment
  
}

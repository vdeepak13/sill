#include <sill/factor/table_factor.hpp>
#include <sill/factor/moment_gaussian.hpp>
#include <sill/factor/canonical_gaussian.hpp>

#include <boost/python.hpp>

#include "converters.hpp"

using namespace boost::python;
using namespace sill;

void def_factor() {
  using self_ns::str;

  // Table factor ========================================================

  double (table_factor::*tab_evala)(const finite_assignment&) const =
    &table_factor::v;
  double (table_factor::*tab_evali)(size_t)                   const =
    &table_factor::v;
  double (table_factor::*tab_evalp)(size_t, size_t)           const =
    &table_factor::v;
  double (table_factor::*tab_logva)(const finite_assignment&) const =
    &table_factor::logv;
  double (table_factor::*tab_logvi)(size_t)                   const =
    &table_factor::logv;
  double (table_factor::*tab_logvp)(size_t, size_t)           const =
    &table_factor::logv;
  void (table_factor::*tab_setva)(const finite_assignment&, double) =
    &table_factor::set_v;
  void (table_factor::*tab_setvi)(size_t, double)                   =
    &table_factor::set_v;
  void (table_factor::*tab_setvp)(size_t, size_t, double)           =
    &table_factor::set_v;
  table_factor (table_factor::*tab_rest)(const finite_assignment&) const =
    &table_factor::restrict;
  table_factor (table_factor::*tab_marg)(const finite_domain&) const =
    &table_factor::marginal;
  table_factor (table_factor::*tab_maxd)(const finite_domain&) const =
    &table_factor::maximum;
  table_factor (table_factor::*tab_mind)(const finite_domain&) const =
    &table_factor::minimum;
  double (table_factor::*tab_maxa)() const =
    &table_factor::maximum;
  double (table_factor::*tab_mina)() const =
    &table_factor::minimum;
  double (table_factor::*tab_ent)() const =
    &table_factor::entropy;
  double (table_factor::*tab_entb)(double base) const =
    &table_factor::entropy;

  typedef std::vector<double> vector_double;
  
  class_<table_factor>("table_factor")
    .def(init<double>())
    .def(init<finite_var_vector>())
    .def(init<finite_var_vector,double>())
    .def(init<finite_var_vector,vector_double>())
    .def("arguments", &table_factor::arguments, return_value_policy<copy_const_reference>())
    .def("size", &table_factor::size)
    .def("arg_vector", &table_factor::arg_vector, return_value_policy<copy_const_reference>())
    .def("__iter__", iterator<dense_table<double> >())
    .def("v", tab_evala)
    .def("v", tab_evali)
    .def("v", tab_evalp)
    .def("logv", tab_logva)
    .def("logv", tab_logvi)
    .def("logv", tab_logvp)
    .def("setv", tab_setva)
    .def("setv", tab_setvi)
    .def("setv", tab_setvp)
    .def(self == self)
    .def("subst_args", &table_factor::subst_args, return_internal_reference<>())
    .def("restrict", tab_rest)
    .def("marginal", tab_marg)
    .def("conditional", &table_factor::conditional)
    .def("is_normalizable", &table_factor::is_normalizable)
    .def("norm_constant", &table_factor::norm_constant)
    .def("normalize", &table_factor::normalize, return_internal_reference<>())
    .def("maximum", tab_maxd)
    .def("minimum", tab_mind)
    .def("maximum", tab_maxa)
    .def("minimum", tab_mina)
    .def("entropy", tab_ent)
    .def("entropy", tab_entb)
    .def("relative_entropy", &table_factor::relative_entropy)
    .def("cross_entropy", &table_factor::cross_entropy)
    .def("mutual_information", &table_factor::mutual_information)
    .def("assignment", &table_factor::assignment)
    .def(self += self)
    .def(self -= self)
    .def(self *= self)
    .def(self /= self)
    .def(self &= self)
    .def(self |= self)
    .def(self += double())
    .def(self -= double())
    .def(self *= double())
    .def(self /= double())
    .def(self + self)
    .def(self - self)
    .def(self * self)
    .def(self / self)
    .def(self & self)
    .def(self | self)
    .def(+self)
    .def(-self)
    .def(self + double())
    .def(double() + self)
    .def(self * double())
    .def(double() * self)
    .def(self - double())
    .def(self / double())
    .def(pow(self, double()))
    .def(str(self));
  
  double (*tab_none)(const table_factor&, const table_factor&) = &norm_1;
  double (*tab_ninf)(const table_factor&, const table_factor&) = &norm_inf;
  double (*tab_none_log)(const table_factor&, const table_factor&) = &norm_1_log;
  double (*tab_ninf_log)(const table_factor&, const table_factor&) = &norm_inf_log;
  table_factor (*tab_inv)(table_factor) = &invert;
  table_factor (*tab_maxp)(const table_factor&, const table_factor&) = &max;
  table_factor (*tab_minp)(const table_factor&, const table_factor&) = &min;
  table_factor (*tab_sum)(const table_factor&, const finite_domain&) = &sum;
  table_factor (*tab_max)(const table_factor&, const finite_domain&) = &max;
  table_factor (*tab_min)(const table_factor&, const finite_domain&) = &min;

  def("norm_1", tab_none);
  def("norm_inf", tab_ninf);
  def("norm_1_log", tab_none_log);
  def("norm_inf_log", tab_ninf_log);
  def("invert", tab_inv);
  def("max", tab_maxp);
  def("min", tab_minp);
  def("sum", tab_sum);
  def("max", tab_max);
  def("min", tab_min);
}

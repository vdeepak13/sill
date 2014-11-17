#include "model.hpp"

#include <sill/factor/table_factor.hpp>
#include <sill/model/bayesian_graph.hpp>
#include <sill/model/markov_graph.hpp>
#include <sill/model/junction_tree.hpp>
#include <sill/model/naive_bayes.hpp>

void def_model() {
  using namespace sill;
  using namespace boost::python;
  
  bayesian_graph_registrar<
    bayesian_graph<finite_variable*, table_factor, table_factor>,
    return_value_policy<copy_const_reference>
  >("bayesian_graph");
  
  markov_graph_registrar<
    markov_graph<finite_variable*, table_factor, table_factor>,
    return_value_policy<copy_const_reference>
  >("markov_graph");

  junction_tree_registrar<
    junction_tree<finite_variable*, table_factor, table_factor>,
    return_value_policy<copy_const_reference>
  >("junction_tree");

  register_iterator_pair<
    directed_graph<finite_variable*, table_factor, table_factor>::vertex_iterator
  >("variable_factor_vertex_range");
  register_iterator_pair<
    directed_graph<finite_variable*, table_factor, table_factor>::neighbor_iterator
  >("variable_factor_neighbor_range");

  typedef naive_bayes<table_factor> nb;

  class_<nb>("naive_bayes")
    .def(init<finite_variable*>())
    .def(init<table_factor>())
    .def(init<table_factor, std::vector<table_factor> >())
    .def("set_prior", &nb::set_prior)
    .def("add_feature", &nb::add_feature)
    .def("label_var", &nb::label_var, return_value_policy<reference_existing_object>())
    .def("features", &nb::features)
    .def("arguments", &nb::arguments)
    .def("prior", &nb::prior)
    .def("feature_cpd", &nb::feature_cpd, return_internal_reference<>())
    .def("__contains__", &nb::contains)
    .def("posterior", &nb::posterior)
    .def("__call__", &nb::operator())
    .def("log_likelihood", &nb::log_likelihood)
    .def("conditional_log_likelihood", &nb::conditional_log_likelihood);

} // def_model

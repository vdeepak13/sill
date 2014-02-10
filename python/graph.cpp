#include "graph.hpp"

#include <sill/graph/directed_graph.hpp>
#include <sill/graph/undirected_graph.hpp>
#include <sill/factor/table_factor.hpp>

void def_graph() {
  using namespace boost::python;
  using namespace sill;

  undirected_graph_registrar<
    undirected_graph<size_t, table_factor, table_factor>,
    return_value_policy<copy_const_reference>
  > ur("undirected_graph");

  directed_graph_registrar<
    directed_graph<size_t, table_factor, table_factor>,
    return_value_policy<copy_const_reference>
  > dr("directed_graph");

  class_<directed_edge<size_t> >("directed_edge_size_t", no_init)
    .def("source", &directed_edge<size_t>::source, return_value_policy<copy_const_reference>())
    .def("target", &directed_edge<size_t>::target, return_value_policy<copy_const_reference>())
    .def(self < self)
    .def(self == self)
    .def(self_ns::str(self));

  class_<undirected_edge<size_t> >("undirected_edge_size_t", no_init)
    .def("source", &directed_edge<size_t>::source, return_value_policy<copy_const_reference>())
    .def("target", &directed_edge<size_t>::target, return_value_policy<copy_const_reference>())
    .def(self < self)
    .def(self == self)
    .def(self_ns::str(self));

  // common iterator ranges
  register_iterator_pair<
    directed_graph<size_t, table_factor, table_factor>::vertex_iterator
  >("size_t_factor_vertex_range");
  register_iterator_pair<
    directed_graph<size_t, table_factor, table_factor>::neighbor_iterator
  >("size_t_factor_neighbor_range");

  // todo: copy constructors
}

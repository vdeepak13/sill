#include "model.hpp"

#include <sill/factor/table_factor.hpp>
#include <sill/model/bayesian_graph.hpp>
#include <sill/model/markov_graph.hpp>
#include <sill/model/junction_tree.hpp>

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

}

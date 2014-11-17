#ifndef SILL_PYTHON_MODEL_HPP
#define SILL_PYTHON_MODEL_HPP

#include <iterator>
#include <vector>

#include "graph.hpp"

// Registers a bayesian_graph template instance
// todo: other mutations from directed_graph?
template <typename Graph, typename PropertyCallPolicy>
struct bayesian_graph_registrar 
  : public basic_graph_registrar<Graph, PropertyCallPolicy> {

  typedef basic_graph_registrar<Graph, PropertyCallPolicy> base;
  typedef typename Graph::vertex vertex;
  typedef typename Graph::node_set node_set;
  typedef typename Graph::edge edge;
  typedef typename Graph::vertex_property vertex_property;
  typedef typename Graph::edge_property edge_property;

  using base::c_;

  bayesian_graph_registrar(const std::string& classname,
                           bool allow_mutate = true)
    : base(classname) {
    using namespace boost::python;

    node_set (Graph::*nodesv)(vertex) const = &Graph::nodes;
    node_set (Graph::*nodese)(edge)   const = &Graph::nodes;
    node_set (Graph::*nodesa)()       const = &Graph::nodes;
    const edge_property&(Graph::*get)(const vertex&, const vertex&) const = 
      &Graph::operator();
    c_.def(init<node_set>())
      .def("parents", &Graph::parents)
      .def("children", &Graph::children)
      .def("__call__", get, PropertyCallPolicy())
      .def("nodes", nodesv)
      .def("nodes", nodese)
      .def("nodes", nodesa)
      .def("ancestors", &Graph::ancestors)
      .def("descendants", &Graph::descendants);
    
    if (allow_mutate) {
      c_.def("add_node", &bayesian_graph_registrar::add_node1)
        .def("add_node", &bayesian_graph_registrar::add_node2)
        .def("add_nodes", &bayesian_graph_registrar::add_nodes1)
        .def("add_nodes", &bayesian_graph_registrar::add_nodes2)
        .def("add_family", &Graph::add_family)
        .def("remove_node", &Graph::remove_node);
    }
  }

private:
  static void add_node1(Graph* graph, vertex v) {
    graph->add_node(v);
  }

  static void add_node2(Graph* graph, vertex v, const vertex_property& vp) {
    graph->add_node(v, vp);
  }

  static void add_nodes1(Graph* graph, const node_set& nodes) {
    graph->add_nodes(nodes);
  }

  static void add_nodes2(Graph* graph, const node_set& nodes, bool allow_dupes) {
    graph->add_nodes(nodes, allow_dupes);
  }
};


// Registers a markov_graph template instance
template <typename Graph, typename PropertyCallPolicy>
struct markov_graph_registrar
  : public basic_graph_registrar<Graph, PropertyCallPolicy> {

  typedef basic_graph_registrar<Graph, PropertyCallPolicy> base;

  typedef typename Graph::vertex vertex;
  typedef typename Graph::edge edge;
  typedef typename Graph::domain_type node_set;
  typedef typename Graph::vertex_property vertex_property;
  typedef typename Graph::edge_property edge_property;
  typedef typename Graph::out_edge_iterator out_edge_iterator;

  using base::c_;

  markov_graph_registrar(const std::string& classname,
                         bool allow_mutate = true)
    : base(classname) {
    using namespace boost::python;

    node_set (Graph::*nodese)(edge) const = &Graph::nodes;
    node_set (Graph::*nodesa)()     const = &Graph::nodes;
    std::pair<out_edge_iterator,out_edge_iterator> (Graph::*edges)(const vertex&) const =
      &Graph::edges;
    bool (Graph::*contains)(const std::set<vertex>&) const =
      &Graph::contains;

    c_.def(init<node_set>())
      .def("nodes", nodese)
      .def("nodes", nodesa)
      .def("neighbors", &Graph::neighbors)
      .def("edges", edges)
      .def("contains", contains)
      .def("__contains__", contains);

    if (allow_mutate) {
      c_.def("add_node", &markov_graph_registrar::add_node1)
        .def("add_node", &markov_graph_registrar::add_node2)
        .def("add_nodes", &markov_graph_registrar::add_nodes1)
        .def("add_nodes", &markov_graph_registrar::add_nodes2)
        .def("remove_node", &Graph::remove_node)
        .def("add_clique", &Graph::add_clique);
    }
  }
  
private:
  static void add_node1(Graph* graph, vertex v) {
    graph->add_node(v);
  }

  static void add_node2(Graph* graph, vertex v, const vertex_property& vp) {
    graph->add_node(v, vp);
  }

  static void add_nodes1(Graph* graph, const node_set& nodes) {
    graph->add_nodes(nodes);
  }

  static void add_nodes2(Graph* graph, const node_set& nodes, bool allow_dupes) {
    graph->add_nodes(nodes, allow_dupes);
  }
};


// Registers a junction_tree template instance
template <typename Graph, typename PropertyCallPolicy>
struct junction_tree_registrar
  : public basic_graph_registrar<Graph, PropertyCallPolicy> {

  typedef basic_graph_registrar<Graph, PropertyCallPolicy> base;

  typedef typename Graph::vertex          vertex;
  typedef typename Graph::edge            edge;
  typedef typename Graph::vertex_property vertex_property;
  typedef typename Graph::edge_property   edge_property;
  typedef typename Graph::node_set        node_set;

  using base::c_;

  junction_tree_registrar(const std::string& classname)
    : base(classname) {
    using namespace boost::python;

    // todo: further constructors and initializers
    bool (Graph::*markedv)(vertex v) const = &Graph::marked;
    bool (Graph::*markede)(edge e) const = &Graph::marked;
    void (Graph::*reachable1)(bool) const = &Graph::compute_reachable;
    void (Graph::*reachable2)(bool, const node_set&) const = &Graph::compute_reachable;

    c_.def(init<const std::vector<node_set>&>())
      //.def(init<const cluster_graph<vertex, vertex_property, edge_property>&>())
      .def("initialize", &junction_tree_registrar::initialize_vc)
      .def("swap", &Graph::swap)
      .def("neighbors", &Graph::neighbors)
      .def("vertex_properties", &Graph::vertex_properties)
      .def("edge_properties", &Graph::edge_properties)
      .def("root", &Graph::root)
      .def("clique", &Graph::clique, return_value_policy<copy_const_reference>())
      .def("separator", &Graph::separator, return_value_policy<copy_const_reference>())
      .def("cliques", &Graph::cliques)
      .def("separators", &Graph::separators)
      .def("marked", markedv)
      .def("marked", markede)
      .def("nodes", &Graph::nodes)
      .def("tree_width", &Graph::tree_width)
      .def("find_clique_cover", &Graph::find_clique_cover)
      .def("find_clique_meets", &Graph::find_clique_meets)
      .def("find_separator_cover", &Graph::find_separator_cover)
      .def("find_intersecting_cliques", &junction_tree_registrar::intersecting_cliques)
      .def("compute_reachable", reachable1)
      .def("compute_reachable", reachable2)
      .def("reachable", &Graph::reachable, return_value_policy<copy_const_reference>())
      .def("mark_subtree_cover", &Graph::mark_subtree_cover)
      .def("check_validity", &Graph::check_validity) // todo: throw exception
      .def("subtree", &Graph::subtree)
      .def("markov_graph", &Graph::markov_graph) // todo: create with potentials
      .def("clear", &Graph::clear)
      .def("remove_vertex", &Graph::remove_vertex)
      .def("merge", &Graph::merge)
      .def("triangulate", &Graph::triangulate);
  }

  // we need this because junction_tree::initialize needs two arguments
  void initialize_vc(Graph* graph, const std::vector<node_set>& cliques) {
    graph->initialize(cliques);
  }

  std::vector<vertex> intersecting_cliques(Graph* graph, const node_set& set) {
    std::vector<vertex> result;
    graph->find_intersecting_cliques(set, std::back_inserter(result));
    return result;
  }
  
};

#endif

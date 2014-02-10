#ifndef SILL_PYTHON_GRAPH_HPP
#define SILL_PYTHON_GRAPH_HPP

#include <set>

#include <boost/python.hpp>
#include <boost/type_traits.hpp>

// in the future, this could be replaced with std::get
template <typename It>
struct iterator_pair {
  static It begin(const std::pair<It,It>& p) { return p.first; }
  static It end(const std::pair<It,It>& p) { return p.second; }
};

// implement string conversion
namespace std {
  template <typename It>
  std::ostream& operator<<(std::ostream& out, const std::pair<It,It>& p) {
    out << "[";
    for (It it = p.first; it != p.second; ++it) {
      if (it != p.first) {
        out << ",";
      }
      out << *it;
    }
    out << "]";
    return out;
  }
}

// defines a class representing the iterator pair for given iterator type
template <typename It>
void register_iterator_pair(const std::string& name) {
  using namespace boost::python;
  using self_ns::str;
  class_<std::pair<It,It> >(name.c_str(), no_init)
    .def("__iter__", range(&iterator_pair<It>::begin,
                           &iterator_pair<It>::end))
    .def(str(self));
}

// defines a return policy that returns the raw pointer value
template <typename Vertex>
struct return_vertex_policy
  : public boost::conditional<
      boost::is_pointer<Vertex>::value,
      boost::python::return_value_policy<boost::python::reference_existing_object>,
      boost::python::default_call_policies
  >::type { };


//! Registers (non-mutating) basic functions & iterators common
//! to all graphs
template <typename Graph, typename PropertyCallPolicy>
struct basic_graph_registrar {
  typedef typename Graph::vertex            vertex;
  typedef typename Graph::edge              edge;
  typedef typename Graph::vertex_property   vertex_property;
  typedef typename Graph::edge_property     edge_property;

  typedef typename Graph::vertex_iterator   vertex_iterator;
  typedef typename Graph::neighbor_iterator neighbor_iterator;
  typedef typename Graph::edge_iterator     edge_iterator;
  typedef typename Graph::in_edge_iterator  in_edge_iterator;
  typedef typename Graph::out_edge_iterator out_edge_iterator;

  boost::python::class_<Graph> c_;

  basic_graph_registrar(const std::string& classname)
    : c_(classname.c_str()) {
    register_basic_iterators(classname);
    register_basic_functions();
  }

  void register_basic_iterators(const std::string& basename) {
    // commented out because these are common types among multiple graph types
    // register_iterator_pair<vertex_iterator>(basename + "_vertex_range");
    // register_iterator_pair<neighbor_iterator>(basename + "_neighbor_range");
    register_iterator_pair<edge_iterator>(basename + "_edge_range");
    register_iterator_pair<in_edge_iterator>(basename + "_in_edge_range");
    register_iterator_pair<out_edge_iterator>(basename + "_out_edge_range");
  }

  void register_basic_functions() {
    using namespace boost::python;

    std::pair<edge_iterator,edge_iterator> (Graph::*edges)() const = &Graph::edges;
    bool (Graph::*contains_vert)(const vertex&) const            = &Graph::contains;
    bool (Graph::*contains_edge)(const edge&) const              = &Graph::contains;
    bool (Graph::*contains_vp)(const vertex&, const vertex&) const = &Graph::contains; 
    const vertex_property& (Graph::*get_vp)(const vertex&) const = &Graph::operator[];
    const edge_property&   (Graph::*get_ep)(const edge&)   const = &Graph::operator[];
    
    c_.def("vertices", &Graph::vertices)
      .def("edges", edges)
      .def("in_edges", &Graph::in_edges)
      .def("out_edges", &Graph::out_edges)
      .def("contains", contains_vert)
      .def("contains", contains_edge)
      .def("contains", contains_vp)
      .def("get_edge", &Graph::get_edge)
      .def("in_degree", &Graph::in_degree)
      .def("out_degree", &Graph::out_degree)
      .def("degree", &Graph::degree)
      .def("empty", &Graph::empty)
      .def("num_vertices", &Graph::num_vertices)
      .def("num_edges", &Graph::num_edges)
      .def("reverse", &Graph::reverse)
      .def("__contains__", contains_vert)
      .def("__contains__", contains_edge)
      .def("__getitem__", get_vp, PropertyCallPolicy())
      .def("__getitem__", get_ep, PropertyCallPolicy())
      .def("null_vertex", &Graph::null_vertex, return_vertex_policy<vertex>())
      .def(self == self)
      .def(self_ns::str(self));
  }
};


//! Registers a graph class directly mutable with graph operations
template <typename Graph, typename PropertyCallPolicy>
struct mutable_graph_registrar 
  : public basic_graph_registrar<Graph, PropertyCallPolicy> {

  typedef basic_graph_registrar<Graph, PropertyCallPolicy> base;
  using base::c_;  

  typedef typename Graph::vertex          vertex;
  typedef typename Graph::edge            edge;
  typedef typename Graph::vertex_property vertex_property;
  typedef typename Graph::edge_property   edge_property;

  mutable_graph_registrar(const std::string& classname)
    : base(classname) {
    void (Graph::*clique)(const std::set<vertex>&) = &Graph::make_clique;
    void (Graph::*clear_ve)(const vertex&)         = &Graph::clear_edges;
    void (Graph::*clear_ae)()                      = &Graph::clear_edges;
    c_.def("add_vertex", &mutable_graph_registrar::add_vertex1)
      .def("add_vertex", &mutable_graph_registrar::add_vertex2)
      .def("add_edge", &mutable_graph_registrar::add_edge1)
      .def("add_edge", &mutable_graph_registrar::add_edge2)
      .def("make_clique", clique)
      .def("__setitem__", &mutable_graph_registrar::set_vp)
      .def("__setitem__", &mutable_graph_registrar::set_ep)
      .def("remove_vertex", &Graph::remove_vertex)
      .def("remove_edge", &Graph::remove_edge)
      .def("clear_edges", clear_ve)
      .def("clear_edges", clear_ae)
      .def("clear", &Graph::clear)
      .def("swap", &Graph::swap);  
  }

private:
  static bool add_vertex1(Graph* graph, const vertex& u) {
    return graph->add_vertex(u);
  }

  static bool add_vertex2(Graph* graph, const vertex& u, const vertex_property& vp) {
    return graph->add_vertex(u, vp);
  }

  static edge add_edge1(Graph* graph, const vertex& u, const vertex& v) {
    return graph->add_edge(u, v).first;
  }

  static edge add_edge2(Graph* graph, const vertex& u, const vertex& v, const edge_property& ep) {
    return graph->add_edge(u, v, ep).first;
  }

  static void set_vp(Graph* graph, const vertex& u, const vertex_property& vp) {
    (*graph)[u] = vp;
  }

  static void set_ep(Graph* graph, const edge& e, const edge_property& ep) {
    (*graph)[e] = ep;
  }
};


//! Registers an undirected_graph template instance
template <typename Graph, typename PropertyCallPolicy>
struct undirected_graph_registrar
  : mutable_graph_registrar<Graph, PropertyCallPolicy> {

  typedef mutable_graph_registrar<Graph, PropertyCallPolicy> base;
  typedef typename Graph::vertex vertex;
  typedef typename Graph::out_edge_iterator out_edge_iterator;

  using base::c_;  

  undirected_graph_registrar(const std::string& classname)
    : base(classname) {
    std::pair<out_edge_iterator,out_edge_iterator> (Graph::*edges)(const vertex&) const =
      &Graph::edges;
    bool (Graph::*contains)(const std::set<vertex>&) const =
      &Graph::contains;
    c_.def("neighbors", &Graph::neighbors)
      .def("edges", edges)
      .def("contains", contains)
      .def("__contains__", contains);
  }
};

//! Registers a directed_graph template instance
template <typename Graph, typename PropertyCallPolicy>
struct directed_graph_registrar
  : mutable_graph_registrar<Graph, PropertyCallPolicy> {

  typedef mutable_graph_registrar<Graph, PropertyCallPolicy> base;
  typedef typename Graph::vertex vertex;
  typedef typename Graph::edge_property edge_property;

  using base::c_;  

  directed_graph_registrar(const std::string& classname)
    : base(classname) {
    bool (Graph::*contains)(const std::set<vertex>&) const =
      &Graph::contains;
    const edge_property&(Graph::*get)(const vertex&, const vertex&) const = 
      &Graph::operator();
    c_.def("parents", &Graph::parents)
      .def("children", &Graph::children)
      .def("contains", contains)
      .def("__contains__", contains)
      .def("__call__", get, PropertyCallPolicy());
  }
};

#endif

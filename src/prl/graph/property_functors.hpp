#ifndef PRL_GRAPH_PROPERTY_FUNCTORS_HPP
#define PRL_GRAPH_PROPERTY_FUNCTORS_HPP

#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_const.hpp>

namespace prl {

  /**
   * A functor type that returns the vertex property by reference.
   * Depending on whether Graph is const-qualified or not,
   * returns a const- or mutable reference to the property.
   * \ingroup graph_types
   */
  template <typename Graph>
  struct vertex_property_t {

    typedef typename Graph::vertex_property vertex_property;

    typedef typename Graph::vertex argument_type;
    typedef typename boost::mpl::if_< 
      boost::is_const<Graph>, const vertex_property&, vertex_property& 
      >::type result_type;

    vertex_property_t(Graph* g_ptr) : g_ptr(g_ptr) { }
    
    result_type operator()(typename Graph::vertex v) const {
      return (*g_ptr)[v];
    }

  private:
    Graph* g_ptr;
  };

  /**
   * A functor type that returns the edge property by reference.
   * Depending on whether Graph is const-qualified or not,
   * returns a const- or mutable reference to the property.
   * \ingroup graph_types
   */
  template <typename Graph>
  struct edge_property_t {
    typedef typename Graph::edge_property edge_property;
    
    typedef typename Graph::edge argument_type;
    typedef typename boost::mpl::if_< 
      boost::is_const<Graph>, const edge_property&, edge_property& 
    >::type result_type;
  
    edge_property_t(Graph* g_ptr) : g_ptr(g_ptr) { }

    result_type operator()(typename Graph::edge e) const {
      return (*g_ptr)[e];
    }
  private:
    Graph* g_ptr;
  };

  //! \relates vertex_property_t
  template <typename Graph>
  vertex_property_t<const Graph> vertex_property_functor(const Graph& graph) {
    return vertex_property_t<const Graph>(&graph);
  }

  //! \relates vertex_property_t
  template <typename Graph>
  vertex_property_t<Graph> vertex_property_functor(Graph& graph) {
    return vertex_property_t<Graph>(&graph);
  }

  //! \relates edge_property_t
  template <typename Graph>
  edge_property_t<const Graph> edge_property_functor(const Graph& graph) {
    return edge_property_t<const Graph>(&graph);
  }

  //! \relates edge_property_t
  template <typename Graph>
  edge_property_t<Graph> edge_property_functor(Graph& graph) {
    return edge_property_t<Graph>(&graph);
  }

}

#endif

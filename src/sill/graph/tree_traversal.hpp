
#ifndef SILL_TREE_TRAVERSAL_HPP
#define SILL_TREE_TRAVERSAL_HPP

#include <deque>
#include <vector>
#include <iterator>

#include <sill/global.hpp>
#include <sill/graph/output_edge_visitor.hpp>
#include <sill/range/reversed.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  namespace impl {

    /**
     * This is a recursive function used to print (a directed subtree
     * of) the junction tree in a human-readable format.  Each vertex
     * is assigned a unique index (on-the-fly), and the written
     * representation uses these indexes to report the topology and
     * the cliques.
     *
     * @param out        the stream to write to
     * @param parent     the parent of the current vertex
     * @param current    the current vertex
     */
    template <typename Graph, typename OutputStream, typename VertexPrinter>
    void print_subtree(const Graph& g,
                       typename Graph::vertex parent,
                       typename Graph::vertex current,
                       OutputStream& out,
                       VertexPrinter vertex_printer) {
      typedef typename Graph::vertex vertex;
      typedef typename Graph::edge edge;
      using namespace std;

      // Write out this node's id and the id's of its neighbors.
      out << "Vertex " << current << ":" << endl;
      out << "  Neighbors: ";

      // Write out all neighbors, parent first
      if (parent != vertex()) out << " " << parent;
      foreach(vertex v, g.neighbors(current))
        if(v != parent) out << " " << v;
      out << endl;

      // Call the vertex printer on this vertex.
      out << "  Information: ";
      vertex_printer(current, g, out);
      out << endl;

      // Call the print function recursively on the children.
      foreach(edge e, g.out_edges(current)) {
        vertex child = e.target();
        if (child != parent) 
          print_subtree(g, current, child, out, vertex_printer);
      }
    }

    /**
     * Struct used by visit_triplets_on_paths().
     */
    template <typename Graph, typename TripletVisitor>
    class TripletToPairVisitor {

      TripletVisitor visitor;

      typename Graph::vertex v;

    public:
      TripletToPairVisitor(TripletVisitor visitor, typename Graph::vertex v)
        : visitor(visitor), v(v) { }

      //! Call TripletVisitor(a, v, b) where v is the fixed vertex.
      void operator()(typename Graph::vertex a, typename Graph::vertex b) {
        visitor(a, v, b);
      }
    }; // class TripletToPair Visitor

  } // namespace sill::impl

  /**
   * Performs a pre-order traversal of a tree (i.e., a singly-connected graph)
   * starting at vertex v.
   * The given edge visitor is applied to each edge during the traversal.
   * The order in which the visitor is applied to the edges is consistent with a
   * breadth-first or depth-first traversal from v.
   *
   * This algorithm requires \f$O(|E|)\f$ time and \f$O(k)\f$ space,
   * where \f$k\f$ is the maximum vertex degree in the graph.
   *
   * @tparam Graph
   *         The graph type.  E.g., undirected_graph fits this type.
   * @tparam EdgeVisitor
   *         A functor type which implements operator()(edge, graph).
   * @param graph
   *        The graph on whose edges the visitor is applied.  This
   *        graph must be singly connected, and it must be either
   *        undirected or bidirectional.
   * @param v
   *        A vertex of the graph.  The distribute protocol is started
   *        at this vertex.
   * @param visitor
   *        The visitor that is applied to each edge of the graph.
   *        (If the graph is biconnected, then this visitor is applied
   *        only to edges directed away from the start vertex.)
   * \ingroup graph_algorithms
   */
  template <typename Graph, typename EdgeVisitor>
  void pre_order_traversal(Graph& graph, 
                           typename Graph::vertex v,
                           EdgeVisitor visitor) {
    typedef typename Graph::vertex vertex;
    typedef typename Graph::edge edge;
    
    // Create a queue of edges that is initialized to contain the
    // outgoing edges of the start vertex.
    std::deque<edge> queue(graph.out_edges(v).first, graph.out_edges(v).second);
    while (!queue.empty()) {
      // Deque the first edge in the queue.
      edge e = queue.front();
      queue.pop_front();
      // Apply the visitor to this edge.
      visitor(e, graph);
      // Enqueue all outgoing edges of the target vertex except the
      // edge leading to the source vertex.
      vertex s = e.source();
      vertex t = e.target();
      foreach(edge e, graph.out_edges(t))
        if (e.target() != s) queue.push_back(e);
    }
  }

  /**
   * Performs a post-order traversal of a tree (i.e., a singly-connected graph)
   * starting at vertex v.
   * The given edge visitor is applied to each edge during the traversal.
   * The reverse of the order in which the visitor is applied to the edges is
   * consistent with a breadth-first or depth-first traversal from v.
   *
   * This algorithm requires \f$O(|E|)\f$ time and space.
   *
   * @tparam Graph
   *         The graph type.  E.g., undirected_graph fits this type.
   * @tparam EdgeVisitor
   *         A functor type which implements operator()(edge, graph).
   * @param graph
   *        The graph on whose edges the visitor is applied.  This
   *        graph must be singly connected, and it must be either
   *        undirected or bidirectional.
   * @param v
   *        A vertex of the graph.  The traversal is started at this
   *        vertex.
   * @param visitor
   *        The visitor that is applied to each edge of the graph.
   *        (If the graph is biconnected, then this visitor is applied
   *        only to edges directed toward from the start vertex.)
   *
   * \ingroup graph_algorithms
   */
  template <typename Graph, typename EdgeVisitor>
  void post_order_traversal(Graph& g,
                            typename Graph::vertex v,
                            EdgeVisitor visitor) {
    typedef typename Graph::edge edge;

    // First compute a pre-order traversal ordering of the edges
    // directed away from v.
    std::vector<edge> edges;
    pre_order_traversal
      (g, v, make_output_edge_visitor(std::back_inserter(edges)));
    // Now visit the reverse of these edges in the reverse order.
    foreach(edge e, make_reversed(edges))
      visitor(g.reverse(e), g);
  }

  /**
   * Visits each (directed) edge of a tree graph once in a traversal
   * such that each \f$v \rightarrow w\f$ is visited after all edges
   * \f$u \rightarrow v\f$ (with \f$u \neq w\f$) are visited.  Orders
   * of this type are said to satisfy the "message passing protocol"
   * (MPP).
   *
   * \ingroup graph_algorithms
   */
  template <typename Graph, typename EdgeVisitor>
  void mpp_traversal(Graph& g,
                     EdgeVisitor visitor,
                     typename Graph::vertex v = typename Graph::vertex()) {
    typedef typename Graph::edge edge;

    // If the start vertex was not specified, choose one arbitrarily.
    if (v == typename Graph::vertex()) {
      if (g.empty()) return;
      v = *g.vertices().first;
    }
    // First compute a pre-order traversal ordering of the edges
    // directed away from v.
    std::vector<edge> edges;
    pre_order_traversal
      (g, v, make_output_edge_visitor(std::back_inserter(edges)));
    // Now visit the reverse of these edges in the reverse order.
    foreach(edge e, sill::make_reversed(edges))
      visitor(g.reverse(e), g);
    // Now visit the edges in the original order.
    foreach(edge e, edges)
      visitor(e, g);
  }

  /**
   * Visits each (directed) edge of a tree graph once in a traversal
   * such that each \f$v \rightarrow w\f$ is visited after all edges
   * \f$u \rightarrow v\f$ (with \f$u \neq w\f$) are visited.  Orders
   * of this type are said to satisfy the "message passing protocol"
   * (MPP).
   *
   * This version of mpp_traversal() uses one EdgeVisitor during
   * the post-order traversal and another EdgeVisitor during the pre-order
   * traversal.  NOTE the post_visitor is called BEFORE the pre_visitor!
   *
   * \ingroup graph_algorithms
   */
  template <typename Graph, typename EdgeVisitorPost, typename EdgeVisitorPre>
  void mpp_traversal(Graph& g,
                     EdgeVisitorPost post_visitor,
                     EdgeVisitorPre pre_visitor,
                     typename Graph::vertex v = typename Graph::vertex()) {
    typedef typename Graph::edge edge;

    // If the start vertex was not specified, choose one arbitrarily.
    if (v == typename Graph::vertex()) {
      if (g.empty()) return;
      v = *g.vertices().first;
    }
    // First compute a pre-order traversal ordering of the edges
    // directed away from v.
    std::vector<edge> edges;
    pre_order_traversal
      (g, v, make_output_edge_visitor(std::back_inserter(edges)));
    // Now visit the reverse of these edges in the reverse order.
    foreach(edge e, sill::make_reversed(edges))
      post_visitor(g.reverse(e), g);
    // Now visit the edges in the original order.
    foreach(edge e, edges)
      pre_visitor(e, g);
  }

  /**
   * Visits all pairs of vertices (i,j) s.t. the path from i to j includes v.
   *
   * If this visits (i,j), it will not visit (j,i); which of the two it visits
   * is arbitrary.
   *
   * @tparam Graph  Undirected graph.
   *
   * \ingroup graph_algorithms
   */
  template <typename Graph, typename Visitor>
  void visit_bracketing_pairs(Graph& g, Visitor visitor,
                              typename Graph::vertex v) {
    typedef typename Graph::vertex vertex;
    typedef typename Graph::edge edge;

    // For each outgoing edge of v, create a list of vertices in that subtree.
    std::vector<std::list<vertex> > subtrees;
    foreach(const edge& e, g.out_edges(v)) {
      std::list<vertex> subtree;
      subtree.push_back(e.target());
      std::deque<edge> queue(1,e);
      while (!queue.empty()) {
        edge e2(queue.front());
        queue.pop_front();
        subtree.push_back(e2.target());
        vertex s = e2.source();
        vertex t = e2.target();
        foreach(edge e3, g.out_edges(t))
          if (e3.target() != s)
            queue.push_back(e3);
      }
      subtrees.push_back(subtree);
    }
    if (subtrees.size() < 2)
      return;
    for (size_t i(0); i < subtrees.size() - 1; ++i) {
      for (size_t j(i+1); j < subtrees.size(); ++j) {
        foreach(const vertex& a, subtrees[i]) {
          foreach(const vertex& b, subtrees[j]) {
            visitor(a, b);
          }
        }
      }
    }
  }

  /**
   * Visits all triplets of vertices (i,j,k) s.t. the path from i to j includes
   * k.
   *
   * If this visits (i,j,k), it will not visit (k,j,i); which of the two it
   * visits is arbitrary.
   *
   * @tparam Graph  Undirected graph.
   */
  template <typename Graph, typename Visitor>
  void visit_triplets_on_paths(Graph& g, Visitor visitor) {
    foreach(const typename Graph::vertex& v, g.vertices()) {
      impl::TripletToPairVisitor<Graph,Visitor> tmp_visitor(visitor, v);
      visit_bracketing_pairs(g, tmp_visitor, v);
    }
  }

  /**
   * Visits all consecutive triplets of vertices (i,j,k) in the given graph.
   *
   * If this visits (i,j,k), it will not visit (k,j,i); which of the two it
   * visits is arbitrary.
   *
   * @tparam Graph  Undirected graph.
   */
  template <typename Graph, typename Visitor>
  void visit_consecutive_triplets(Graph& g, Visitor visitor) {
    typedef typename Graph::vertex vertex;
    foreach(const vertex& v, g.vertices()) {
      if (g.degree(v) > 1) {
        std::vector<vertex>
          neighbors(g.neighbors(v).first, g.neighbors(v).second);
        for (size_t i(0); i < neighbors.size() - 1; ++i) {
          for (size_t j(i+1); j < neighbors.size(); ++j) {
            visitor(neighbors[i], v, neighbors[j]);
          }
        }
      }
    }
  }

  /**
   * Prints a human-readable representation of the tree to the
   * supplied output stream.
   * \ingroup graph_algorithms
   */
  template <typename Graph,
            typename OutputStream,
            typename VertexPrinter>
  void print_tree(const Graph& g, OutputStream& out,
                  VertexPrinter vertex_printer) {
    if(!g.empty())
      impl::print_subtree(g, typename Graph::vertex(), *g.vertices().first,
                          out, vertex_printer);
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_TREE_TRAVERSAL_HPP

#ifndef SILL_DIRECTED_EDGE_HPP
#define SILL_DIRECTED_EDGE_HPP

#include <boost/functional/hash.hpp>

#include <sill/global.hpp>

namespace sill {

  template<typename V, typename VP, typename EP> 
  class directed_graph;

  template<typename V, typename VP, typename EP> 
  class directed_multigraph;

  /**
   * A class that represents a directed edge.
   * \ingroup graph_types
   */
  template <typename Vertex>
  class directed_edge {
  private:
    //! Vertex from which the edge originates
    Vertex m_source;

    //! Vertex to which the edge emenates
    Vertex m_target;

    /**
     * The property associated with this edge.  Edges maintain a private
     * pointer to the associated property.  However, this pointer can only
     * be accessed through the associated graph. This permits graphs to
     * return iterators over edges and permits constant time lookup for
     * the corresponding edge properties. The property is stored as a void*,
     * to simplify the type of the edges.
     */ 
    void* m_property;

    /**
     * directed_graphs are made friends here to have access to the internal
     * edge property.  
     */
    template <typename V, typename VP, typename EP>
    friend class directed_graph;

    template <typename V, typename VP, typename EP>
    friend class directed_multigraph;


  public:
    //! Default constructor
    directed_edge() : m_source(), m_target(), m_property() {}

    //! Constructor which permits setting the edge_property
    directed_edge(Vertex source, Vertex target, void* property = NULL)
      : m_source(source), m_target(target), m_property(property) { }

    bool operator<(const directed_edge& other) const {
      return (m_source < other.m_source) || 
        (m_source == other.m_source && m_target < other.m_target);
    }

    bool operator==(const directed_edge& other) const {
      return m_source == other.m_source && m_target == other.m_target;
    }

    bool operator!=(const directed_edge& other) const {
      return !operator==(other);
    }

    const Vertex& source() const {
      return m_source;
    }

    const Vertex& target() const {
      return m_target;
    }

  }; // class directed_edge

  //! \relates directed_edge
  template <typename Vertex>
  std::ostream& operator<<(std::ostream& out, const directed_edge<Vertex>& e) {
    out << e.source() << " --> " << e.target();
    return out;
  }

  //! \relates directed_edge
  template <typename Vertex>
  inline size_t hash_value(const directed_edge<Vertex>& e) {
    return boost::hash_value(std::make_pair(e.source(), e.target()));
  }
  
} // namespace sill

#endif


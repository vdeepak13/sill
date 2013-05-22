
#ifndef SILL_LEARNING_DECOMPOSABLE_CHANGE_HPP
#define SILL_LEARNING_DECOMPOSABLE_CHANGE_HPP

#include <sill/model/decomposable.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Class for representing a unit change to a decomposable model.
   * This is structured as:
   *  - vertex_checks: list of checks to make sure the move is still valid
   *  - vertex_changes: list of changes to vertices (including the creation of
   *     new vertices)
   *  - insert_edges1/2/3: list of edges to create
   *  - (deleted edges): These are not specified explicitly but are computed
   *     at commit time.  Note new vertices won't have deleted edges.
   *     Also note that the same number of edges must be deleted/inserted
   *     to maintain a tree; if an edge has an empty separator, this fact can
   *     be used to determine if the edge should be deleted.
   *
   * Note: The MarginalPtr is meant to be flexible.  It must implement
   *       operator*(), which returns a marginal whose arguments are a superset
   *       of the corresponding vertex's domain.
   *  - By default, it is a shared pointer to a marginal.
   *    This allows similar moves which are spawned from each other
   *    to share marginals.
   *  - It could be a class which overloads operator*() to, e.g., 
   *    look up the marginal in a statistics class which may have
   *    pre-computed marginals or may compute the marginal from data.
   *
   * @param F            type of factor used in the decomposable model
   * @param MarginalPtr  MarginalPtr::operator*() must return a marginal over
   *                     at least the variables in the corresponding vertex.
   */
  template <typename F, typename MarginalPtr = boost::shared_ptr<F> >
  class decomposable_change {

  public:
    //! The type of factor used in the decomposable model
    typedef F factor_type;

    //! The domain type of the factor
    typedef typename F::domain_type domain_type;

    //! The type of vertex associated with the model
    typedef typename decomposable<F>::vertex vertex;

    //! The type of edge associated with the model
    typedef typename decomposable<F>::edge edge;

    // RIGHT HERE NOW: Do I need to add in edge checks?

    /**
     * Class for validating a change in a decomposable model.
     */
    class vertex_check {
    public:
      const vertex vertex_ID;
      const domain_type required_variables;
      //! True iff the vertex must have exactly the required variables
      const bool required_variables_exact;
      //! Variables required in neighbors before the change
      const domain_type required_neighbor_variables;
      //! Variables not allowed in neighbors before the change
      const domain_type disallowed_neighbor_variables;
      vertex_check(const vertex& vertex_ID,
                 const domain_type& required_variables,
                 const bool required_variables_exact,
                 const domain_type& required_neighbor_variables,
                 const domain_type& disallowed_neighbor_variables)
        : vertex_ID(vertex_ID), required_variables(required_variables),
          required_variables_exact(required_variables_exact),
          required_neighbor_variables(required_neighbor_variables),
          disallowed_neighbor_variables(disallowed_neighbor_variables) {
      }
    }; // class vertex_check

    /**
     * Class for representing a change to a vertex in a decomposable model.
     */
    class vertex_change {
    public:
      //! True iff this represents the creation of a new vertex
      const bool new_vertex;
      //! Valid if new_vertex == false
      const vertex vertex_ID;
      //! If new_vertex == true, this is ignored.
      const domain_type delete_variables;
      //! If new_vertex == true, this specifies the clique.
      const domain_type insert_variables;
      //! marginal after change (not required if vertex is being deleted)
      const MarginalPtr marginal_ptr;

      vertex_change(const bool new_vertex, const vertex& vertex_ID,
                    const domain_type& delete_variables,
                    const domain_type& insert_variables,
                    const MarginalPtr& marginal_ptr)
        : new_vertex(new_vertex), vertex_ID(vertex_ID),
          delete_variables(delete_variables),
          insert_variables(insert_variables), marginal_ptr(marginal_ptr) {
      }
    }; // class vertex_change

    //! Vertex checks to see if the move is valid
    std::vector<vertex_check> vertex_checks;
    //! Vertex changes required for move
    std::vector<vertex_change> vertex_changes;
    //! Insert edges between old vertices
    std::vector<edge> insert_edges1;
    //! Insert edges between old-new vertex pairs
    //! The new vertices are indexed by their index in the vertex_changes
    //! vector.
    std::vector<std::pair<vertex, size_t> > insert_edges2;
    //! Insert edges between new vertices
    //! The new vertices are indexed by their index in the vertex_changes
    //! vector.
    std::vector<std::pair<size_t, size_t> > insert_edges3;

  }; // class decomposable_change

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DECOMPOSABLE_CHANGE_HPP

#ifndef SILL_FACTOR_GRAPH_HPP
#define SILL_FACTOR_GRAPH_HPP

#include <sill/global.hpp>
#include <sill/factor/concepts.hpp>
#include <sill/graph/bipartite_graph.hpp>

#include <vector>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * This class represents a factor graph model. A factor graph is a bipartite
   * graph, where type-1 vertices correspond to variables, and type-2 vertices
   * correspond to factors, indexed by size_t. There is an undirected edge
   * between a variable and a factor if the variable is in the domain of the
   * factor. This model represents (an unnormalized) distribution over the
   * variables as a product of all the contained factors.
   *
   * \ingroup model
   * \tparam F the factor type stored in this model
   */
  template <typename F>
  class factor_graph
    : public bipartite_graph<typename F::variable_type*, size_t, F> {
    concept_assert((Factor<F>));

    // Public type declarations
    // =========================================================================
  public:
    typedef typename F::variable_type   variable_type;
    typedef typename F::domain_type     domain_type;
    typedef typename F::assignment_type assignment_type;
    typedef typename F::record_type     record_type;

    // bring the inherited types up
    typedef bipartite_graph<variable_type*, size_t, F> base;
    typedef typename base::vertex1_iterator vertex1_iterator;
    typedef typename base::vertex2_iterator vertex2_iterator;

    // bring functions from base
    using base::print_degree_distribution;
    
    // Public functions
    //==========================================================================
  public:
    //! Creates an empty factor graph
    factor_graph() : next_id_(1) { }

    //! Swaps two factor graphs in constant time
    void swap(factor_graph& other) {
      base::swap(other);
      std::swap(next_id_, other.next_id_);
    }

    //! Returns the range of all variables in the graph (alias for vertices1)
    std::pair<vertex1_iterator, vertex1_iterator>
    variables() const {
      return this->vertices1();
    }

    //! Returns the range of all factor ids in the graph (alias for vertices2)
    std::pair<vertex2_iterator, vertex2_iterator>
    factor_ids() const {
      return this->vertices2();
    }

    //! Returns the number of variables in the graph (alias for num_vertices1)
    size_t num_variables() const {
      return this->num_vertices1();
    }

    //! Returns the number of factors in the graph (alias for num_vertices2)
    size_t num_factors() const {
      return this->num_vertices2();
    }

    //! Returns the factor with the given id. The vertex id must be present.
    const F& factor(size_t id) const {
      return (*this)[id];
    }

    //! Returns the arguments of the factor with the given id
    const domain_type& cluster(size_t id) const {
      return (*this)[id].arguments();
    }

    //! Prints the degree distribution to the given stream
    void print_degree_distribution(std::ostream& out) const {
      out << "Degree distribution of variables:" << std::endl;
      print_degree_distribution(out, variables());
      out << "Degree distribution of factors:" << std::endl;
      print_degree_distribution(out, factor_ids());
    }

    /**
     * Adds a factor to the factor graph.
     * \returns the corresponding id or 0 if the factor has no arguments
     */
    size_t add_factor(const F& f) {
      if (f.arguments().empty()) {
        return 0; // not added
      }
      while (this->contains(next_id_)) {
        ++next_id_;
      }
      size_t id = next_id_++;
      this->add_vertex(id, f);
      foreach (variable_type* var, f.arguments()) {
        this->add_edge(var, id);
      }
      return id;
    }

    /**
     * Updates the factor associated with the given id, modifying the
     * graph structure to reflect the new factor arguments.
     */
    void update_factor(size_t id, const F& f) {
      // if the factor is a constant, just drop the factor
      if (f.arguments().empty()) {
        this->remove_vertex(id);
        return;
      }
      const domain_type& cluster = this->cluster(id);
      // remove the edges to the dropped variables
      foreach (variable_type* var, cluster) {
        if (!f.arguments().count(var)) {
          this->remove_edge(var, id);
        }
      }
      // add edges to the new variables
      foreach (variable_type* var, f.arguments()) {
        if (!cluster.count(var)) {
          this->add_edge(var, id);
        }
      }
      // set the factor
      factor(id) = f;
    }

    /**
     * Normalize all factors
     */
    void normalize() {
      foreach (size_t id, factor_ids()) {
        factor(id).normalize();
      }
    }

    /**
     * Condition on an assignment
     */
    void condition(const assignment_type& a) {
      foreach (typename assignment_type::const_reference p, a) {
        variable_type* var = p.first;
        if (!this->contains(var)) {
          continue;
        }
        foreach (size_t id, this->neighbors(var)) {
          if (cluster(id).count(var)) { // not processed yet
            update_factor(id, factor(id).restrict(a));
          }
        }
        this->remove_vertex(var);
      }
    }

    /**
     * Simplify the model by merging factors. For each factor f(X),
     * if a factor g(Y) exists such that X \subseteq Y, the factor
     * g is multiplied by f, and f is removed from the model.
     */
    void simplify() {
      std::vector<size_t> ids(factor_ids().first, factor_ids().second);
      foreach(size_t f_id, ids) {
        const domain_type& f_args = cluster(f_id);
        // identify the variable with the fewest connected factors
        size_t min_degree = std::numeric_limits<size_t>::max();
        variable_type* var = NULL;
        foreach(variable_type* v, f_args) {
          if (this->degree(v) < min_degree) {
            min_degree = this->degree(v);
            var = v;
          }
        }
        // identify a subsuming factor among the new neighbors of var
        foreach(size_t g_id, this->neighbors(var)) {
          if (f_id != g_id && includes(cluster(g_id), f_args)) {
            factor(g_id) *= factor(f_id);
            this->remove_vertex(f_id);
            break;
          }
        }
      }
    }

  private:
    //! Returns the factor with the given id. The factor must be present.
    F& factor(size_t id) {
      return (*this)[id];
    }

    //! The next id that will be used when inserting new factors
    size_t next_id_;

  }; // class factor_graph

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

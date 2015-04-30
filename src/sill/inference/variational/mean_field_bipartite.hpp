#ifndef SILL_MEAN_FIELD_BIPARTITE_HPP
#define SILL_MEAN_FIELD_BIPARTITE_HPP

#include <sill/global.hpp>
#include <sill/boost_unordered_utils.hpp>
#include <sill/factor/traits.hpp>
#include <sill/graph/bipartite_graph.hpp>
#include <sill/parallel/worker_group.hpp>

#include <boost/unordered_map.hpp>

#include <functional>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A class that runs the mean field algorithm for a bipartite graph.
   * The computation is performed synchronously, first for all type-1
   * vertices and then for all type-2 vertices. The number of worker
   * threads is controlled by a parameter to the constructor.
   * 
   * \tparam Vertex1 the type that represents a type-1 vertex
   * \tparam Vertex2 the type that represents a type-2 vertex
   * \tparam NodeF the factor type associated with vertices
   * \tparam EdgeF the factor type associated with edges
   */
  template <typename Vertex1, typename Vertex2,
            typename NodeF, typename EdgeF = NodeF>
  class mean_field_bipartite {
  public:
    static_assert(same_argument_type<NodeF, EdgeF>::value,
                  "The node & edge factor types must have the same argument type");
    static_assert(same_real_type<NodeF, EdgeF>::value,
                  "The node & edge factor types must have the same real type");

    // factor-related typedefs
    typedef typename NodeF::real_type               real_type;
    typedef typename NodeF::probability_factor_type belief_type;

    // graph-related typedefs
    typedef bipartite_graph<Vertex1, Vertex2, NodeF, NodeF, EdgeF> model_type;
    typedef typename graph_type::vertex_type vertex_type;
    typedef typename graph_type::edge_type   edge_type;
    
    /**
     * Creates a mean field engine for the given graph.
     * The graph vertices must not change after initialization
     * (the potentials may).
     *
     * \param num_threads the number of worker threads
     */
    explicit mean_field_bipartite(const model_type* model,
                                  size_t num_threads = 1)
      : model_(*model), num_threads_(num_threads) {
      foreach(Vertex1 v, model_.vertices1()) {
        beliefs1_[v] = belief_type(model_[v].arguments()).normalize();
      }
      foreach(Vertex2 v, model_.vertices2()) {
        beliefs2_[v] = belief_type(model_[v].arguments()).normalize();
      }
    }

    /**
     * Performs a single iteration of mean field.
     */
    real_type iterate() {
      real_type diff1 = update_all(model_.vertices1());
      real_type diff2 = update_all(model_.vertices2());
      return (diff1 + diff2) / model_.num_vertices();
    }

    /**
     * Returns the belief for a type-1 vertex.
     */
    const belief_type& belief(Vertex1 v) const {
      return get(beliefs1_, v);
    }

    /**
     * Returns the belief for a type-2 vertex.
     */
    const belief_type& belief(Vertex2 v) const {
      return get(beliefs2_, v);
    }

    /**
     * Returns the belief for a vertex.
     */
    const belief_type& belief(vertex_type v) const {
      return v.type1() ? belief(v.v1()) : belief(v.v2());
    }

  private:
    /**
     * Updates the given range of vertices and returns the sum of the
     * factor differences.
     * \tparam It forward iterator over vertices to update
     */
    template <typename It>
    real_type update_all(std::pair<It, It> vertices) {
      typedef typename std::iterator_traits<It>::value_type vertex_type;
      if (num_threads_ > 1) {
        worker_group<vertex_type, real_type> workers(
          num_threads_,
          boost::bind(&mean_field_bipartite::update<vertex_type>, this, _1),
          std::plus<real_type>()
        );
        workers.enqueue_all(vertices);
        workers.join();
        return workers.aggregate_result();
      } else {
        real_type sum = 0.0;
        foreach (vertex_type v, vertices) {
          sum += update(v);
        }
        return sum;
      }
    }

    /**
     * Updates a single vertex.
     * \tparam Vertex the vertex type
     */
    template <typename Vertex>
    real_type update(Vertex v) {
      NodeF result = model_[v];
      foreach (edge_type e, model_.in_edges(v)) {
        model_[e].exp_log_multiply(belief(e.source()), result);
      }
      belief_type new_belief(result);
      new_belief.normalize();
      swap(const_cast<belief_type&>(belief(v)), new_belief);
      return sum_diff(new_belief, belief(v));
    }
    
    //! The underlying graphical model
    const model_type& model_;

    //! The number of worker threads
    size_t num_threads_;

    //! A map of current beliefs for type-1 vertices
    boost::unordered_map<Vertex1, belief_type> beliefs1_;

    //! A map of current beliefs for type-2 vertices
    boost::unordered_map<Vertex2, belief_type> beliefs2_;

  };

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

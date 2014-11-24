#ifndef SILL_BETHE_GBP_HPP
#define SILL_BETHE_GBP_HPP

#include <vector>
#include <algorithm>
#include <map>

#include <sill/global.hpp>
#include <sill/factor/concepts.hpp>
#include <sill/factor/invalid_operation.hpp>
#include <sill/graph/directed_graph.hpp>
#include <sill/model/interfaces.hpp>
#include <sill/datastructure/set_index.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A belief propagation engine that uses a Bethe approximation.
   *
   * \ingroup inference
   */
  template <typename F>
  class bethe_bp {
    concept_assert((Factor<F>));
    
    // Public type declarations
    //==========================================================================
  public:
    //! The underlying factor type
    typedef F factor_type;

    //! The type of variables used by the factor
    typedef typename F::variable_type variable_type;

    //! The type of the factor's domain
    typedef typename F::domain_type domain_type;

    //! The type of vertices used in the message graph
    typedef size_t vertex;
    
    //! The type of edges used in the message graph
    typedef directed_edge<size_t> edge;

    // Protected data members
    //==========================================================================
  protected:
    //! A bipartite graph that stores the node potentials on the nodes 
    //! and messages on the directed edges
    directed_graph<size_t, F, F> g;
    
    //! An index that speeds up the factor look-ups
    set_index<domain_type, size_t> domain_index;

    //! An index that speeds up the look-up of individual variables
    std::map<variable_type*, size_t> var_index;

    //! The latest vertex id
    size_t id;

    // Construction
    //==========================================================================
  public:
    //! Constructs the engine for a graphical model.
    bethe_bp(const factorized_model<F>& model) {
      id = 1;

      // Add the vertices for each variable
      foreach(variable_type* v, model.arguments()) {
        g.add_vertex(id, F(make_domain(v), 1));
        var_index[v] = id;
        domain_index.insert(make_domain(v), id);
        id++;
      }

      // Add a vertex for each factor
      foreach(const F& factor, model.factors())
        add_factor(factor);
    }

    //! Constructs an engine with the specified partitioning of the variables
    bethe_bp(const factorized_model<F>& model,
             const std::vector<domain_type>& clusters) {
      id = 1;
      domain_type args = model.arguments();

      // Add a vertex for each cluster
      foreach(const domain_type& cluster, clusters) {
        assert(includes(args, cluster));
        args = set_difference(args, cluster);
        g.add_vertex(id, F(cluster, 1));
        foreach(variable_type* v, cluster)
          var_index.insert(std::make_pair(v, id));
        domain_index.insert(cluster, id);
        id++;
      }
      assert(args.empty());
      
      // Add a vertex for each factor
      foreach(const F& factor, model.factors())
        add_factor(factor);
    }

    virtual ~bethe_bp() { }

    //! Adds a factor to the engine.
    virtual size_t add_factor(const F& factor) {
      g.add_vertex(id, factor);
      domain_index.insert(factor.arguments(), id);
      
      // add the edges
      foreach(variable_type* v, factor.arguments()) {
        assert(var_index.count(v) > 0);
        size_t u = var_index[v];
        domain_type args = set_intersect(factor.arguments(), g[u].arguments());
        g.add_edge(id, u, F(args, 1));
        g.add_edge(u, id, F(args, 1));
      }

      return id++;
    }
    
    // Iteration
    //==========================================================================
    //! Computes the message along an edge
    F compute_message(edge e) const {
      size_t u = e.source();
      size_t v = e.target();
//       std::cerr << g[u].arguments() << " --> " << g[v].arguments() 
//                 << std::endl;
      F msg = g[u];
      foreach(edge ein, g.in_edges(u))
        if (ein.source() != v) 
          msg *= g[ein];
      //std::cerr << msg << std::endl;
      //msg *= g[u];
      try {
        msg = msg.marginal(g[v].arguments());
      } catch(invalid_operation& e) {
        msg = F(set_intersect(msg.arguments(), g[v].arguments()), 1);
      }
      //std::cerr << msg << std::endl;
      msg.normalize();
      return msg;
    }

    //! Passes message between two vertices
    double pass_flow(edge e, double eta = 1) {
      F msg = compute_message(e);
      double residual = norm_inf(g[e], msg);
      g[e] = (eta == 1) ? msg : weighted_update(g[e], msg, eta);
      return residual;
    }

    //! Performs one round of updates with the given learning rate
    virtual double iterate(double eta = 1) = 0;

    // Queries
    //==========================================================================
    //! Returns the graph with potentials & messages
    const directed_graph<size_t, F, F>& graph() const {
      return g;
    }

    //! Returns the messages
    const F& message(edge e) const {
      return g[e];
    }

    //! Returns the estimated belief for a given vertex in the message graph
    F belief(size_t v) const {
      F result = g[v];
      foreach(edge e, g.in_edges(v))
        result *= g[e];
      result.normalize();
      return result;
    }

    //! Returns the estimated belief for a subset of variables
    //! \param vars a set of variables. must be covered by some cluster.
    F belief(const domain_type& vars) const {
      size_t v = domain_index.find_min_cover(vars);
      assert(v); // a cover was found
      return belief(v).marginal(vars);
    }
    
    //! Returns a collection of marginals over the individual variables
    std::vector<F> node_beliefs() const {
      typedef std::pair<variable_type*, size_t> variable_vertex_pair;
      std::vector<F> beliefs;
      foreach(variable_vertex_pair p, var_index) 
        beliefs.push_back(belief(p.second));
      return beliefs;
    }

  }; // class bethe_bp

  
} // namespace sill

#include <sill/macros_undef.hpp>

#endif

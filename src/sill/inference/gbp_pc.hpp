#ifndef SILL_GBP_HPP
#define SILL_GBP_HPP

#include <algorithm> // for std::max
#include <set>

#include <boost/scoped_ptr.hpp>
#include <boost/unordered_map.hpp>

#include <sill/factor/concepts.hpp>
#include <sill/factor/norms.hpp>
#include <sill/factor/invalid_operation.hpp>
#include <sill/model/interfaces.hpp>
#include <sill/model/region_graph.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A class that implements the parent-to-child generalized belief 
   * propagation algorithm.
   *
   * @tparam F A type that implements the Factor concept.
   *
   * \ingroup inference
   */
  template <typename F>
  class gbp_pc {
    concept_assert((Factor<F>));

    // Public types
    //==========================================================================
  public:
    typedef typename F::variable_type variable_type;
    typedef typename F::domain_type   domain_type;
    typedef directed_edge<size_t>     edge;

    // Protected members
    //==========================================================================
  protected:
    //! A map type that holds messages
    typedef boost::unordered_map<edge, F> message_map_type;

    //! The underlying region graph
    region_graph<variable_type*, F> graph;

    //! The factor norm
    boost::scoped_ptr< factor_norm<F> > norm;

    //! The messages
    message_map_type msg;
    
    //! The edges that are used to compute the belief over a region
    mutable boost::unordered_map<size_t, std::vector<edge> > belief_edges;

    //! The edges in the numerator of the message
    boost::unordered_map<edge, std::vector<edge> > message_edges_numerator;

    //! The edges in the denominator of the message
    boost::unordered_map<edge, std::vector<edge> > message_edges_denominator;

    // Public functions
    //==========================================================================
  public:
    gbp_pc(const region_graph<variable_type*, F>& graph)
      : graph(graph), norm(new factor_norm_inf<F>()) {
      initialize_messages();
      initialize_factors();
      initialize_belief_edges();
      initialize_message_edges();
    }

    virtual ~gbp_pc() { }

    //! Initializes the messages to 1
    void initialize_messages() {
      foreach(edge e, graph.edges()) {
        assert(includes(graph.cluster(e.source()), graph.cluster(e.target())));
        domain_type args = graph.cluster(e.target());
        message(e) = F(args, 1);
      }
    }

    //! Initializes the factors to 1
    void initialize_factors() {
      foreach(size_t v, graph.vertices()) 
        graph[v] = F(graph.cluster(v), 1);
    }

    //! Initializes the factors to the given model
    void initialize_factors(const factorized_model<F>& model) {
      initialize_factors();
      foreach(const F& factor, model.factors()) {
        size_t v = graph.find_root_cover(factor.arguments());
        assert(v);
        graph[v] *= factor;
//         using namespace std;
//         cerr << "Factor " << factor.arguments() 
//              << " placed at " << graph.cluster(v) << endl;
      }
    }

    //! Pre-computes which messages contribute to the belief.
    void initialize_belief_edges() {
      foreach(size_t u, graph.vertices()) {
        std::vector<edge> edges;
        std::set<size_t> descendants_u = graph.descendants(u);
        descendants_u.insert(u);
        // messages from external sources to regions in down+(u)
        foreach(size_t v, descendants_u) {
          foreach(edge e, graph.in_edges(v))
            if (!descendants_u.count(e.source()))
              edges.push_back(e);
        }
        belief_edges[u] = edges;
//         using namespace std;
//         cerr << "Belief at region " << u << ": edges " << edges << endl;
      }
    }

    //! Precomputes which message contribute to a message
    void initialize_message_edges() {
      foreach(edge e_msg, graph.edges()) {
        std::vector<edge> edges;
        size_t u = e_msg.source();
        size_t v = e_msg.target();
        std::set<size_t> descendants_u = graph.descendants(u);
        std::set<size_t> descendants_v = graph.descendants(v);
        descendants_u.insert(u);
        descendants_v.insert(v);

        // numerator: edges from sources external to u that are outside 
        // the scope of influence of v
        foreach(size_t w, descendants_u)
          if (!descendants_v.count(w)) {
            foreach(edge e, graph.in_edges(w))
              if (!descendants_u.count(e.source()))
                edges.push_back(e);
          }
        message_edges_numerator[e_msg] = edges;
//         using namespace std;
//         cerr << "Message " << e_msg << ": numerator edges " << edges << endl;

        // denominator: information passed from u to regions below v indirectly
        edges.clear();
        foreach(size_t w, descendants_v) {
          foreach(edge e, graph.in_edges(w))
            if (e != e_msg &&
                descendants_u.count(e.source()) &&
                !descendants_v.count(e.source()))
              edges.push_back(e);
        }
        message_edges_denominator[e_msg] = edges;
//         using namespace std;
//         cerr << "Message " << e_msg << ": denominator edges " << edges << endl;

      }
    }
    

    //! Performs one iteration
    virtual double iterate(double eta) = 0;

    //! Returns a belief for a region
    F belief(size_t v) const {
      F result = graph[v];
      foreach(edge e, belief_edges[v])
        result *= message(e);
      return result.normalize();
    }

    //! Returns the marginal over a set of variables
    F belief(const domain_type& vars) const {
      size_t v = graph.find_cover(vars); 
      assert(v);
      return belief(v).marginal(vars);
    }

    //! Returns the marginal over a single variable
    F belief(variable_type* var) const {
      return belief(make_domain(var));
    }


    // Implementation
    //==========================================================================
  protected:
    //! Returns the message from region u to region v
    F& message(edge e) {
      return msg[e];
    }

    //! Returns the message from region u to region v
    const F& message(edge e) const {
      typename message_map_type::const_iterator it = msg.find(e);
      assert(it != msg.end());
      return it->second;
    }

    //! Passes a message along an edge
    double pass_message(edge e_msg, double eta) {
      size_t u = e_msg.source();
      size_t v = e_msg.target();
      F msg = graph[u];
      foreach(edge e, message_edges_numerator[e_msg])
        msg *= message(e);
      foreach(edge e, message_edges_denominator[e_msg])
        msg /= message(e);
      try {
        msg = msg.marginal(graph.cluster(v)).normalize();
      } catch(invalid_operation& e) {
        std::cerr << ".";
        msg = F(graph.cluster(v), 1);
      }
      
      // compute the residual and update the message
      double residual = (*norm)(msg, message(e_msg));
      message(e_msg) =
        (eta == 1) ? msg : weighted_update(message(e_msg), msg, eta);

      return residual;
    }

  }; // class gbp_pc

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

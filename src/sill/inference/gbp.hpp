#ifndef SILL_GBP_HPP
#define SILL_GBP_HPP

#include <algorithm> // for std::max

#include <boost/scoped_ptr.hpp>
#include <boost/unordered_map.hpp>

#include <sill/factor/concepts.hpp>
#include <sill/factor/norms.hpp>
#include <sill/factor/invalid_operation.hpp>
#include <sill/model/interfaces.hpp>
#include <sill/model/region_graph.hpp>
#include <sill/base/stl_util.hpp>
#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A class that implements generalized synchronous generalized belief
   * propagation.
   *
   * @tparam F A type that implements the Factor concept.
   *
   * \ingroup inference
   */
  template <typename F>
  class gbp {
    concept_assert((Factor<F>));

    // Public types
    //==========================================================================
    typedef typename F::variable_type variable_type;
    typedef typename F::domain_type   domain_type;

    // Protected members
    //==========================================================================
  protected:
    //! A map type that holds messages
    typedef boost::unordered_map<std::pair<size_t, size_t>, F> message_map_type;

    //! The underlying region graph
    region_graph<variable_type*, F> graph;

    //! The factor norm
    boost::scoped_ptr< factor_norm<F> > norm;

    //! The pseudo message (does not account for counting numbers)
    message_map_type pseudo_msg;

    //! The message
    message_map_type msg;

    // Public functions
    //==========================================================================
  public:
    gbp(const region_graph<variable_type*, F>& graph)
      : graph(graph), norm(new factor_norm_inf<F>()) {
      initialize_messages();
      initialize_factors();
    }

    //! Initializes the messages to 1
    void initialize_messages() {
      foreach(directed_edge<size_t> e, graph.edges()) {
        domain_type args = set_intersect(graph.cluster(e.source()), 
                                         graph.cluster(e.target()));
        pseudo_message(e.source(), e.target()) = F(args, 1);
        pseudo_message(e.target(), e.source()) = F(args, 1);
        message(e.source(), e.target()) = F(args, 1);
        message(e.target(), e.source()) = F(args, 1);
      }
    }

    //! Initializes the factors to 1
    void initialize_factors() {
      foreach(size_t v, graph.vertices()) {
        graph[v] = F(1);
//         using namespace std;
//         cerr << graph.cluster(v) << ": " << beta(v) << endl;
      }
    }

    //! Initializes the factors to the given model
    void initialize_factors(const factorized_model<F>& model) {
      initialize_factors();
      foreach(const F& factor, model.factors()) {
        size_t v = graph.find_root_cover(factor.arguments());
        assert(v);
        graph[v] *= factor;
      }
    }

    //! Performs one iteration
    virtual double iterate(double eta) = 0;

    //! Returns the belief for a region
    F belief(size_t v) const {
      F result = pow(graph[v], graph.counting_number(v));
      foreach(size_t u, graph.parents(v))
        result *= message(u, v);
      foreach(size_t u, graph.children(v))
        result *= message(u, v);

      return result.normalize();
    }

    //! Returns the marginal over a set of variables
    F belief(const domain_type& vars) const {
      size_t v = graph.find_cover(vars); 
      assert(v);
      return belief(v).marginal(vars);
    }

    //! Returns the marginal over a single variable
    F belief(finite_variable* var) const {
      return belief(make_domain(var));
    }

    // Implementation
    //==========================================================================
  protected:
    //! Returns a pseudo-message from region u to region v
    F& pseudo_message(size_t u, size_t v) {
      return pseudo_msg[std::make_pair(u, v)];
    }

    //! Returns the message from region u to region v
    F& message(size_t u, size_t v) {
      return msg[std::make_pair(u, v)];
    }

    //! Returns the message from region u to region v
    const F& message(size_t u, size_t v) const {
      typename message_map_type::const_iterator it =
        msg.find(std::make_pair(u,v));
      assert(it != msg.end());
      return it->second;
    }

    //! Passes a message from region u to region v
    //! u and v must be adjacent
    double pass_message(size_t u, size_t v, double eta) {
      // compute beta for the child
      double br = graph.contains(u, v) ? beta(v) : beta(u);

      // compute the pseudo message (this is m0 in the Yedidia paper)
      F m0 = pow(graph[u], graph.counting_number(u));
      foreach(size_t w, graph.parents(u)) 
        if (w != v) m0 *= message(w, u);
      foreach(size_t w, graph.children(u))
        if (w != v) m0 *= message(w, u);      
      try {
        pseudo_message(u, v) = m0.marginal(graph.cluster(v)).normalize();
      } catch(std::invalid_argument& e) {
        using namespace std;
        cerr << "c_r = " << graph.counting_number(u) << endl;
        cerr << "beta = " << br << endl;
        cerr << m0 << endl;
        cerr << pow(graph[u], graph.counting_number(u)) << endl;
        foreach(size_t w, graph.parents(u)) 
          if (w != v) cerr << message(w, u);
        foreach(size_t w, graph.children(u))
          if (w != v) cerr << message(w, u);
        assert(false);
      } catch(invalid_operation& e) {
        std::cerr << ".";
        pseudo_message(u, v) = 
          F(set_intersect(graph.cluster(u), graph.cluster(v)), 1);
      }
      
      // compute the true message
      F new_msg;
      new_msg  = pow(pseudo_message(u, v), br);
      new_msg *= pow(pseudo_message(v, u), br - 1);
      // new_msg = new_msg.normalize();
      // this causes a bug in pre-C++11 OS X STL
      new_msg.normalize();
//       foreach(double& x, new_msg.values()) 
//         if (x != x) {
//           using namespace std;
//           new_msg = 1;
//           pseudo_message(u, v) = 1;
//           pseudo_message(v, u) = 1;
//           cerr << ".";
//           break;
//           cerr << br << endl;
//           cerr << new_msg << endl;
//           cerr << pseudo_message(u, v) << endl;
//           cerr << pseudo_message(v, u) << endl;
//           cerr << pow(pseudo_message(u, v),br) << endl;
//           cerr << pow(pseudo_message(v, u),br-1) << endl;
//           assert(false);
//         }

      //std::cerr << br << " ";

      // compute the residual and update the message
      F oldmessage = message(u, v);
      F& msg = message(u, v);
      msg = (eta == 1) ? new_msg : weighted_update(msg, new_msg, eta);
      double residual = (*norm)(msg, oldmessage);
      return residual;
      // check
    }

    //! Returns the beta coefficient for vertex v
    double beta(size_t v) const {
      if (graph.in_degree(v) > 0) {
        double qr = double(1 - graph.counting_number(v)) / graph.in_degree(v);
        double br = 1.0 / (2 - qr);
        assert(qr != 2);
        return br;
      } else
        return 1;
    }

  }; // class gbp

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

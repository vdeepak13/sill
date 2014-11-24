#ifndef SILL_BELIEF_PROPAGATION_HPP
#define SILL_BELIEF_PROPAGATION_HPP

#include <vector>
#include <algorithm>
#include <list>
#include <map>

#include <boost/function.hpp>
#include <boost/random/lagged_fibonacci.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/exponential_distribution.hpp>

#include <sill/factor/norms.hpp>
#include <sill/factor/commutative_semiring.hpp>
#include <sill/global.hpp>

#include <sill/datastructure/mutable_queue.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! \addtogroup inference
  //! @{

  /**
   * An engine that performs loopy belief propagation.
   * If the underlying markov network changes, the results are undefined.
   * The lifetime of the Markov network object must extend past the lifetime
   * of this object.
   */
  template < typename GM >
  class loopy_bp_engine {
    // BUGBUG: WE SHOULD CONCEPT CHECK G IS A GRAPH

  public:
    typedef typename GM::factor_type F;
    typedef typename F::domain_type domain_type;

    typedef typename GM::vertex vertex;
    typedef typename GM::edge edge;
    typedef std::pair<vertex, vertex> vertex_pair;

  protected:
    //! A collection of messages
    typedef std::map<vertex_pair, F> message_map;

    //! A reference to the markov network used in the computations
    //! \todo could also be a shared_ptr
    const GM& gm;

    //! A map that stores the messages (mutable since we may insert defaults)
    mutable message_map msg;

    //! The commutative semiring used by the engine
    //! \todo Make this a unique_ptr
    boost::shared_ptr<commutative_semiring<F> > csr;

    //! The norm used to evaluate the change in messages
    factor_norm<F>& norm;

    //! The functor used to initialize the messages
    boost::function<F(vertex)> init_functor;

    //! The total number of updates applied (possibly fewer than computed)
    unsigned long n_updates;

    //! Computes the message from one node to another
    F compute_message(edge e) const {
      vertex u = e.source(), v = e.target();
      F f = gm[u];
      foreach(vertex w, gm.adjacent_vertices(u)) {
        if (w != v) csr->combine_in(f, message(w, u));
      }
      return csr->collapse(csr->combine(f, gm[e]), make_domain(v)).normalize();
    }

    //! Passes flow from one node to another
    double pass_flow(edge e, double eta=1) {
      F new_message = compute_message(e);
      double residual = norm(message(e), new_message);
      message(e) = (eta == 1) ?
        new_message : weighted_update(message(e), new_message, eta);
      n_updates++;
      return residual;
    }

  public:

    //! Resets all the messages using the init_functor, if present
    virtual void reset() {
      foreach(vertex v, gm.vertices()) {
        foreach(edge e, gm.in_edges(v)) {
          if (init_functor) {
            message(e) = init_functor(v).normalize();
          } else {
            message(e) = F(make_domain(v), 1).normalize();
          }
        }
      }
    }

    /**
     * Constructs a loopy bp engine for the given graph.
     * @param gm      graphical model
     * @param norm    norm used to compute residuals
     */
    loopy_bp_engine(const GM& gm, const factor_norm<F>& norm)
      : gm(gm),
        csr(new sum_product<F>()),
        norm(*norm.clone()),
        n_updates(0) {
      reset();
    }

    virtual ~loopy_bp_engine() { delete &norm; }

    //! Returns the network that we perform inference over
    const GM& graphical_model() const { return gm; }

    //! Sets the engine to use the specified commutative semiring
    void set_csr(commutative_semiring<F>* csr) {
      this->csr.reset(csr);
    }

    //! Sets the init functor
    void set_init_functor(const boost::function<F(vertex)>& fnc) {
      init_functor = fnc;
    }

    /**
     * This method should be implemented by all BP engines and should.
     * complete a single iteration of BP
     */
    virtual double iterate1(double eta) = 0;

    //! Performs a given number of iterations
    double iterate(size_t n, double eta = 1) {
      using std::max;
      double residual = 0;
      for(size_t i = 0; i < n; i++)
        residual = max(residual, iterate1(eta));
      return residual;
    }

    //! The number of updates performed so far
    unsigned long num_updates() const { 
      return n_updates; 
    }

    //! Returns a message.
    //! If not already present, the message is default-initialized.
    F& message(vertex from, vertex to) {
      return msg[vertex_pair(from, to)];
    }

    F& message(edge e) {
      return message(e.source(), e.target());
    }

    const F& message(vertex from, vertex to) const {
      return msg[vertex_pair(from,to)];
    }

    const F& message(edge e) const {
      return message(e.source(), e.target());
    }

    //! Computes the node belief
    F belief(vertex u) const {
      F f = gm[u];
      foreach(vertex v, gm.adjacent_vertices(u)) {
        f *= message(v, u);
      }
      return f.normalize();
    }


    //! Computes the edge belief (is this correct?)
    F belief(edge e) const {
      vertex u = e.source(), v = e.target();
      F fu = gm[u], fv = gm[v];
      foreach(vertex w, gm.adjacent_vertices(u))
        if (w != v) fu *= message(w, u);
      foreach(vertex w, gm.adjacent_vertices(v))
        if (w != u) fv *= message(w, v);
      return (gm[e] * fu * fv).normalize();
    }

    //! Returns the pseudo-marginals for all the nodes
    std::list<F> node_beliefs() const {
      std::list<F> marginals;
      foreach(vertex v, gm.vertices())
        marginals.push_back(belief(v));
      return marginals;
    }

    //! Returns the residual for the given directed edge
    virtual double residual(edge e) const {
      return norm(compute_message(e), message(e));
    }

    //! Average residual
    double average_residual() const {
      double result = 0;
      foreach(edge e, gm.edges()) {
        result += residual(e);
        result += residual(gm.reverse(e));
      }
      return result / gm.num_edges();
    }

    //! Max residual
    double max_residual() const {
      double result = 0;
      foreach(edge e, gm.edges()) {
        double res = residual(e) + residual(gm.reverse(e));
        if (res > result)
          result = res;
      }
      return result;
    }

    //! The residual type
    double expected_residual(double n = 1) const {
      using std::pow;
      double num = 0, denom = 0, r;
      foreach(edge e, gm.edges()) {
        edge er = gm.reverse(e);
        r = residual(e); num += std::pow(r, n+1); denom += std::pow(r, n);
        r = residual(er); num += std::pow(r, n+1); denom += std::pow(r, n);
      }
      if (denom > 0) return num / denom; else return 0;
    }

  };

  /**
   * Loopy BP engine that updates the messages synchronously
   * Optionally, performs randomization
   */
  template <typename GM>
  class synchronous_loopy_bp : public loopy_bp_engine<GM> {
    typedef loopy_bp_engine<GM> base;

  public:
    typedef typename base::F F;  // Factor type
    typedef typename base::edge edge;
    typedef typename base::vertex vertex;
    typedef typename base::vertex_pair vertex_pair;

  protected:
    // shortcuts from the base
    using base::compute_message;
    using base::message;
    using base::norm;
    using base::gm;
    using base::msg;
    using base::n_updates;

    //! The new messages
    typename base::message_map newmsg;

    //! The exponent that determines the probability of an update
    //! The message is updated with probability norm(m,m_old)^exponent
    double exponent;

    //! The source of randomness
    boost::mt19937 generator;
    boost::uniform_real<double> uniform01;

    //! Passes flow from one node to another
    double pass_flow(edge e, double eta=1) {
      using std::pow;
      F new_message = compute_message(e);
      double residual = norm(message(e), new_message);
      if (exponent == 0 || uniform01(generator) < std::pow(residual, exponent)) {
        newmsg[vertex_pair(e.source(), e.target())] = (eta == 1) ?
          new_message : weighted_update(message(e), new_message, eta);
        n_updates++;
      }
      return residual;
    }

  public:
    /**
     * Constructs a synchronous loopy bp engine for the given graph.
     * @param gm       graphical model
     * @param exponent The exponent that determines the probability of an
     *                 update: The message is updated with probability
     *                 norm(m,m_old)^exponent.
     * @param norm     norm used to compute residuals
     */
    synchronous_loopy_bp(const GM& gm,
                         double exponent = 0,
                         const factor_norm<F>& norm = factor_norm_inf<F>())
      : base(gm, norm), exponent(exponent) { }

    double iterate1(double eta) {
      using std::max;
      double residual = 0;
      foreach(edge e, gm.edges()) {
        residual = max(residual, pass_flow(e, eta));
        residual = max(residual, pass_flow(gm.reverse(e), eta));
      }
      msg = newmsg;
      return residual;
    }
  };

  /**
   * Loopy BP engine that updates the messages in a round-robin manner
   */
  template <typename GM>
  class asynchronous_loopy_bp : public loopy_bp_engine<GM> {
    typedef loopy_bp_engine<GM> base;

  public:
    typedef typename base::F F;
    typedef typename base::edge edge;
    typedef typename base::vertex vertex;

  protected:
    // shortcuts from the base
    using base::pass_flow;
    using base::gm;

  public:
    /**
     * Constructs an asynchronous loopy bp engine for the given graph.
     * @param gm      graphical model
     * @param norm    norm used to compute residuals
     */
    asynchronous_loopy_bp(const GM& gm,
                          const factor_norm<F>& norm = factor_norm_inf<F>())
      : base(gm, norm) { }

    double iterate1(double eta) {
      using std::max;
      double residual = 0;
      foreach(edge e, gm.edges()) {
        residual = max(residual, pass_flow(e, eta));
        residual = max(residual, pass_flow(gm.reverse(e), eta));
      }
      return residual;
    }
  };

  /**
   * Loopy BP engine that updates the messages according to the
   * the largest change in their neighbors.
   * @param F Factor type
   * @param norm_type A function object type that computes the difference
   *                    between two messages
   */
  template <typename GM>
  class residual_loopy_bp : public loopy_bp_engine<GM> {

    typedef loopy_bp_engine<GM> base;
  public:
    typedef typename base::F F;
    typedef typename base::edge edge;
    typedef typename base::vertex vertex;
    typedef typename base::vertex_pair vertex_pair;

  protected:
    // shortcuts from the base
    using base::compute_message;
    using base::pass_flow;
    using base::gm;
    using base::message;
    using base::norm;

    mutable_queue<vertex_pair, double> q; //< The queue of weights

    //! Updates the residuals for an edge
    void update_residual(edge e) {
      vertex_pair vp(e.source(), e.target());
      double r = norm(message(e), compute_message(e));
      if (!q.contains(vp)) q.push(vp, r); else q.update(vp, r);
    }

  public:
    /**
     * Resets all messages of BP engine.
     */
    void reset() {
      base::reset();
      foreach(edge e, gm.edges()) {
        update_residual(e);
        update_residual(gm.reverse(e));
      }
    }

    /**
     * Constructs a residual loopy bp engine for the given graph.
     * @param gm      graphical model
     * @param norm    norm used to compute residuals
     */
    residual_loopy_bp(const GM& gm,
                      const factor_norm<F>& norm = factor_norm_inf<F>())
      : base(gm, norm) {
      // Pass the flow along each directed edge
      foreach(edge e, gm.edges()) {
        pass_flow(e);
        pass_flow(gm.reverse(e));
      }
      // Compute the residuals
      foreach(edge e, gm.edges()) {
        update_residual(e);
        update_residual(gm.reverse(e));
      }
    }

    double iterate1(double eta) {
      using namespace std;
      if (!q.empty()) {
        // extract the leading candidate edge
        vertex_pair vp; double r;
        boost::tie(vp, r) = q.pop();
        vertex u = vp.first, v = vp.second;
        edge e = gm.get_edge(u, v);

        // pass the flow and update dependent messages
        // cout << "Passing flow between " << vp << " r=" << r << endl;
        double residual = pass_flow(e, eta);
        foreach(edge e_update, gm.out_edges(v)) {
          if (e_update.target() != u) update_residual(e_update);
        }
        if (eta<1) update_residual(e);
        return residual;
      } else return 0;
    }

    double residual(edge e) const {
      try {
        return q.get(std::make_pair(e.source(), e.target()));
      } catch(std::out_of_range exc) {
        // double r2 = base::residual(e);
        // return r2;
        return 0;
      }
    }

  };


  /**
   * Loopy BP engine that updates the messages according to a random number
   * generated from a random distribution with lambda = residual.
   * @param F Factor type
   * @param norm_type A function object type that computes the difference
   *                    between two messages
   */
  template <typename GM>
  class exponential_loopy_bp : public loopy_bp_engine<GM> {
    typedef loopy_bp_engine<GM> base;
  public:
    typedef typename base::F F;
    typedef typename base::edge edge;
    typedef typename base::vertex vertex;
    typedef typename base::vertex_pair vertex_pair;

  protected:
    // shortcuts from the base
    using base::compute_message;
    using base::norm;
    using base::pass_flow;
    using base::gm;
    using base::message;

    //! The exponent of the residual that affects how close we get to the max
    double exponent;

    //! The queue of 1/time
    mutable_queue<vertex_pair, double> q;

    //! The time of the latest updated message
    double current_time;

    boost::lagged_fibonacci607 rng;

    //! Updates the residuals for an edge
    void update_residual(edge e) {
      using namespace boost;
      using std::pow;
      vertex_pair vp(e.source(), e.target());
      double r = norm(message(e), compute_message(e));
      if (r > 0) {
        double t = current_time +
          exponential_distribution<double>(std::pow(r, exponent))(rng);
        if (!q.contains(vp)) q.push(vp, 1/t); else q.update(vp, 1/t);
      }
    }

  public:

    /**
     * Resets all messages of BP engine.
     */
    void reset() {
      base::reset();
      foreach(edge e, gm.edges()) {
        update_residual(e);
        update_residual(gm.reverse(e));
      }
    }

    /**
     * Constructs an exponential loopy bp engine for the given graph.
     * @param gm       graphical model
     * @param exponent The exponent of the residual that affects how close we
     *                 get to the max
     * @param norm     norm used to compute residuals
     */
    exponential_loopy_bp(const GM& gm,
                         double exponent = 1,
                         const factor_norm<F>& norm = factor_norm_inf<F>())
      : base(gm, norm), exponent(exponent), current_time(0) {
      foreach(edge e, gm.edges()) {
        update_residual(e);
        update_residual(gm.reverse(e));
      }
    }

    double iterate1(double eta) {
      using namespace std;
      if (!q.empty()) {
        // extract the leading candidate edge
        vertex_pair vp; double r;
        boost::tie(vp, r) = q.pop();
        vertex u = vp.first, v = vp.second;
        edge e = gm.get_edge(u, v);
        current_time = 1/r;

        // pass the flow and update dependent messages
        // cout << "Passing flow between " << vp << " r=" << r << endl;
        double residual = pass_flow(e, eta);
        foreach(edge e_update, gm.out_edges(v)) {
          if (e_update.target() != u) update_residual(e_update);
        }
        if (eta<1) update_residual(e);
        return residual;
      } else return 0;
    }
  };

  //! @}

} // namespace sill


#include <sill/macros_undef.hpp>

#endif

#ifndef SILL_LEARNT_DECOMPOSABLE_HPP
#define SILL_LEARNT_DECOMPOSABLE_HPP

#include <sill/model/decomposable.hpp>
#include <sill/model/learnt_junction_tree.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // #define SILL_VERBOSE // for debugging

  /**
   * A version of the decomposable model class which is designed to be learnt
   * from a dataset rather than pre-specified.  In the decomposable class,
   * when the structure of the model is changed, the new marginals are
   * computed using the distribution given in the model itself; however,
   * in the learnt_decomposable class, the new marginals are specified by
   * the caller (using, e.g., empirical marginals).
   *
   * Note: This allows unsafe access to the decomposable model, permitting more
   * efficient operations such as changes to cliques, edges, and marginals.
   *
   * Note: To ensure the stability of descriptors and iterators
   *       after copying, use the supplied custom copy constructor.
   *
   * @tparam F a type that models the DistributionFactor concept
   *
   * \ingroup model
   */
  template <typename F>
  class learnt_decomposable : public decomposable<F>
  {
    concept_assert((DistributionFactor<F>));

    // Public type declarations
    // =========================================================================
  public:
    //! The base class
    typedef decomposable<F> base;

    // Import types from base class
    typedef typename base::factor_type factor_type;
    typedef typename base::variable_type variable_type;
    typedef typename base::domain_type domain_type;
    typedef typename base::assignment_type assignment_type;
    typedef typename base::vertex vertex;
    typedef typename base::edge edge;
    typedef typename base::vertex_property vertex_property;
    typedef typename base::edge_property edge_property;

    /**
     * PARAMETERS
     *  - maximal_JTs (bool): Indicates if model should always have maximal-size
     *     cliques.  This should be enforced by the caller.
     *     (default = false)
     *  - max_clique_size (size_t): Maximal clique size (>0) in model.
     *     This should be enforced by the caller.
     *     (default = 2)
     */
    class parameters {
    private:
      bool maximal_JTs_;
      size_t max_clique_size_;
    public:
      parameters() : maximal_JTs_(false), max_clique_size_(2) { }
      parameters& maximal_JTs(bool value) {
        maximal_JTs_ = value; return *this;
      }
      parameters& max_clique_size(size_t value) {
        assert(value > 0);
        max_clique_size_ = value; return *this;
      }
      bool maximal_JTs() const { return maximal_JTs_; }
      size_t max_clique_size() const { return max_clique_size_; }
    }; // class parameters

    // Protected type declarations and data members
    // =========================================================================
  protected:

    //! This mutable junction tree may be modified and then swapped
    //! in constant time with the tree stored by the base class.
    learnt_junction_tree<variable_type*, F, F> learnt_jt;

    using base::args;
    using base::jt;

    //! Maximum clique size in model.
    //! This must be enforced by the caller.
    size_t max_clique_size_;

    //! If true, then this search only allows maximal junction trees
    //! (JTs with maximal-size cliques).
    //! This must be enforced by the caller.
    bool maximal_JTs_;

    // Constructors
    // =========================================================================
  public:

    /**
     * Default constructor. The distribution has no arguments and
     * is identically one.
     */
    learnt_decomposable(parameters params = parameters())
      : max_clique_size_(params.max_clique_size()),
        maximal_JTs_(params.maximal_JTs()) { }

    /**
     * Initializes the learnt_decomposable model to the given set of clique
     * marginals.  The marginals must be triangulated.
     */
    template <typename FactorRange>
    explicit learnt_decomposable(const FactorRange& factors,
                                 parameters params = parameters(),
                                 typename FactorRange::iterator* = 0)
      : base(factors), max_clique_size_(params.max_clique_size()),
        maximal_JTs_(params.maximal_JTs()) { }

    learnt_decomposable(const std::vector<F>& factors,
                        parameters params = parameters());

    /**
     * Initializes the learnt_decomposable model to a given decomposable model.
     */
    explicit learnt_decomposable(const decomposable<F>& model,
                                 parameters params = parameters())
      : base(model), max_clique_size_(params.max_clique_size()),
        maximal_JTs_(params.maximal_JTs()) { }

    // Accessors
    // =========================================================================

    //! Maximum clique size in model.
    //! This must be enforced by the caller.
    size_t max_clique_size() const { return max_clique_size_; }

    //! If true, then this search only allows maximal junction trees
    //! (JTs with maximal-size cliques).
    //! This must be enforced by the caller.
    bool maximal_JTs() const { return maximal_JTs_; }

    // Mutating functions which are not in decomposable
    // =========================================================================

    //! Add a new clique with the given potential and no edges.
    //! The caller is responsible for creating edges as necessary.
    vertex add_clique(const domain_type& d, const F& f) {
      assert(d == f.arguments());
      args = set_union(args, d);
      learnt_jt.swap(jt);
      vertex v(learnt_jt.add_clique(d, f));
      learnt_jt.swap(jt);
      return v;
    }

    //! Set an existing clique to have the given domain and potential.
    //! This updates the adjacent edges' separators and potentials.
    void set_clique(vertex u, const domain_type& d, const F& f) {
      assert(d == f.arguments());
      args = set_union(args, d);
      learnt_jt.swap(jt);
      learnt_jt.set_clique(u, d);
      learnt_jt[u] = f;
      foreach(edge e, learnt_jt.out_edges(u)) {
        learnt_jt[e] = f.marginal(learnt_jt.separator(e));
      }
      learnt_jt.swap(jt);
    }

    //! Insert edge e into the model, collapsing either clique potential
    //! to compute the edge potential.
    //! @todo Add an option to pass in the potential.
    //! @return new edge descriptor
    edge add_edge(edge e) {
      learnt_jt.swap(jt);
      edge new_e(learnt_jt.add_edge(e.source(), e.target()));
      F f(learnt_jt[e.source()]);
      learnt_jt[new_e] = f.marginal(learnt_jt.separator(new_e));
      learnt_jt.swap(jt);
      return new_e;
    }

    //! Insert edge <v1,v2> into the model, collapsing either clique potential
    //! to compute the edge potential.
    //! @todo Add an option to pass in the potential.
    //! @return new edge descriptor
    edge add_edge(vertex v1, vertex v2) {
      learnt_jt.swap(jt);
      edge new_e(learnt_jt.add_edge(v1, v2));
      F f(learnt_jt[v1]);
      learnt_jt[new_e] = f.marginal(learnt_jt.separator(new_e));
      learnt_jt.swap(jt);
      return new_e;
    }

    //! Removes an edge from the model.
    void remove_edge(edge e) {
      learnt_jt.swap(jt);
      learnt_jt.remove_edge(e);
      learnt_jt.swap(jt);
    }

    //! Removes edge <v1,v2> from the model.
    void remove_edge(vertex v1, vertex v2) {
      learnt_jt.swap(jt);
      learnt_jt.remove_edge(v1, v2);
      learnt_jt.swap(jt);
    }

    //! Clear all nodes and arguments from the model.
    void clear() {
      args.clear();
      jt.clear();
    }

  }; // class learnt_decomposable

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNT_DECOMPOSABLE_HPP

#ifndef PRL_DECOMPOSABLE_FRAGMENT_HPP
#define PRL_DECOMPOSABLE_FRAGMENT_HPP

#include <stack>
#include <map>

#include <boost/serialization/vector.hpp>

#include <prl/global.hpp>
#include <prl/factor/factor.hpp>
#include <prl/factor/prior_likelihood.hpp>
#include <prl/model/decomposable.hpp>
#include <prl/range/joined.hpp>

#include <prl/macros_def.hpp>

namespace prl {
  
  /**
   * Decomposable fragment is a factor that represents a subset 
   * of marginals of the probability distribution. Under certain
   * conditions, the edges of the true model can be identified 
   * locally from the cliques of the factor, which can be used
   * to perform marginalization operations.
   *
   * Implements: Factor
   *
   * @tparam F the prior factor type. The factor type must support 
   *          sum-product CSR operations.
   *
   * @tparam G the likelihood factor type.
   *
   * \ingroup factor_types
   * \see Factor
   */
  template <typename F, typename G = F>
  class decomposable_fragment : public factor {
    concept_assert((DistributionFactor<F>));
    concept_assert((DistributionFactor<G>));
    
    // Public type declarations
    //==========================================================================
  public:
    //! The storage type of the underlying factor
    typedef typename F::result_type result_type;

    //! implements Factor::domain_type
    typedef typename F::domain_type domain_type;

    //! implements Factor::variable_type
    typedef typename F::variable_type variable_type;

    //! the assignment type of the factor
    typedef std::map<variable_type*, typename variable_type::value_type> 
      assignment_type;

    //! implements Factor::record_type
    typedef typename F::record_type record_type;

    //! The result of a collapse operation
    typedef decomposable_fragment collapse_type;

    //! implements Factor::collapse_ops
    static const unsigned collapse_ops = 1 << sum_op;

    //! implements Factor::combine_ops
    static const unsigned combine_ops = 1 << product_op;

    // Private data members
    //==========================================================================
  private:

    //! The type that stores factors in this fragment
    typedef std::vector<prior_likelihood<F,G> > pl_vector;
    
    //! The type of the constituent factor
    typedef prior_likelihood<F, G> factor_type;

    //! The argument set of this factor
    domain_type args;

    // The collection of factors that forms this fragment
    copy_ptr<pl_vector> pls;

    // Constructors and conversion operators
    //==========================================================================
  public:
    //! Serialize members
    void save(oarchive& ar) const {
      ar << args << *pls;
    }

    //! Deserialize the members
    void load(iarchive& ar) {
      ar >> args >> *pls;
    }

    //! Default constructor
    decomposable_fragment() : pls(new pl_vector()) { }

  #ifndef SWIG
    //! Singleton constructor
    decomposable_fragment(const prior_likelihood<F,G>& pl) 
      : args(pl.arguments()), pls(new pl_vector(1, pl)) { }
  #endif

    //! Converts a prior to a decomposable fragment. 
    //! Care must be taken when F == G that decomposable_fragment is not
    //! constructed with a likelihood alone.
    decomposable_fragment(const F& prior)
      : args(prior.arguments()), pls(new pl_vector(1, prior)) { }

    //! Conversion constructor
    decomposable_fragment(const constant_factor& factor)
      : pls(new pl_vector(1, factor)) { }

    //! Constructs a model fragment factor 
    //! @param Range a collection of prior marginals or prior-likelihoods.
    template <typename FactorRange>
    explicit decomposable_fragment(const FactorRange& factors) {
      concept_assert((InputRangeConvertible<FactorRange, factor_type>));
      pls.reset(new pl_vector(boost::begin(factors), boost::end(factors)));
      args.clear();
      foreach(const domain_type &a, cliques()) {
        args = set_union(args, a);
      }
    }

    //! Construct a model fragment factor with given arguments
    //! @param Range a collection of prior marginals or prior-likelihoods
    template <typename FactorRange>
    decomposable_fragment(const FactorRange& factors, const domain_type& args) 
      : args(args) {
      pls.reset(new pl_vector(boost::begin(factors), boost::end(factors)));
    }

  #ifdef SWIG
    decomposable_fragment(const std::vector<F>& priors);
  #endif
      
    //! Conversion to a constant factor. \todo For now, always returns 1
    operator constant_factor() const {
      assert(arguments().empty());
      return 1;
    }

    //! Converts this factor to the prior type; equivalent to calling flatten()
    operator F() const {
      return flatten();
    }

    //! Converts this decomposable fragment to a decomposable model.
    operator decomposable<F>() const {
      return to_decomposable();
    }

    //! Converts to human-readable format
    operator std::string() const {
      std::ostringstream out; out << *this; return out.str(); 
    }

    // Accessors
    //==========================================================================
    //! Returns the argument set of this factor
    const domain_type& arguments() const {
      return args;
    }

    //! Returns the number of prior-likelihood factors in this fragment
    size_t size() const {
      return pls->size();
    }

  #ifndef SWIG    
    //! Returns the collection of prior-likelihoods, contained in this fragment
    const pl_vector& factors() const {
      return *pls;
    }
  #endif

    //! Returns the collection of priors, contained in this fragment
    forward_range<const F&> priors() const {
      return make_transformed(*pls, std::mem_fun_ref(&factor_type::prior));
    }
    
    //! Returns the collection of likelihoods, contained in this fragment
    forward_range<const G&> likelihoods() const {
      return make_transformed(*pls,std::mem_fun_ref(&factor_type::likelihood));
    }

    //! Returns the cliques, contained in this fragment
    forward_range<const domain_type&> cliques() const {
      return make_transformed(*pls, std::mem_fun_ref(&factor_type::arguments));
    }

    //! Returns true if two decomposable fragments are equal
    //! (i.e., have the same arguments and the same list of components).
    bool operator==(const decomposable_fragment& other) const {
      assert(pls);
      assert(other.pls);
      return arguments() == other.arguments() && 
        (pls == other.pls || *pls == *other.pls);
    }

    //! Returns true if two decomposable fragments are not equal
    bool operator!=(const decomposable_fragment& other) const {
      return !operator==(other);
    }

    //! Returns true if this factor precedes the other in the lexicographical
    //! ordering
    bool operator<(const decomposable_fragment& other) const {
      assert(pls);
      assert(other.pls);
      if (arguments() != other.arguments())
        return arguments() < other.arguments();
      else
        return prl::lexicographical_compare(*pls, *other.pls);
    }

    //! Converts this decomposable fragment to a decomposable model.
    //! \param thin if true, thins the decomposable model as necessary
    decomposable<F> to_decomposable(bool thin = true) const {
      decomposable<F> d(priors());
      if(thin) /*d.thin()*/; else d.check_running_intersection();
      d *= likelihoods();
      return d;
    }

    /**
     * Returns a flat representation of this factor.
     * The prior marginals must be calibrated.
     * \param force If true, marginalizes out variables from cliques as 
     *        necessary to form a decomposable model
     */
    F flatten(bool thin = true) const {
      return to_decomposable(thin).marginal(arguments());
    }

    // Factor operations
    //==========================================================================
    //! implements Factor::operator()
    result_type operator()(const assignment_type& a) const {
      assert(false);
      return 0;
    }

    //! implements Factor::combine_in for multiplication
    decomposable_fragment&
    combine_in(const decomposable_fragment& x, op_type op) {
      check_supported(op, combine_ops);
      append(*pls, x.factors());
      args.insert(x.arguments());
      return *this;
    }

    //! implements Factor::combine_in for multiplication
    decomposable_fragment&
    combine_in(const G& likelihood, op_type op) {
      check_supported(op, combine_ops);
      foreach(factor_type& pl, *pls) {
        if (includes(pl.arguments(), likelihood.arguments())) {
          pl *= likelihood;
          return *this;
        }
      }
      throw std::invalid_argument("The arguments of the likelihood are not covered by any prior");
    }

    //! implements Factor::combine_in for multiplication
    decomposable_fragment&
    combine_in(const constant_factor& other, op_type op) {
      check_supported(op, combine_ops);
      assert(!pls->empty());
      pls->front() *= other;
      return *this;
    }

    //! implements Factor::collapse for summation
    decomposable_fragment collapse(const domain_type& retain, op_type op) const{
      check_supported(op, collapse_ops);
      if (includes(retain, arguments())) return *this; // not much to do
      
      // Construct the canonical tree
      // Each vertex of the tree is associated with a PL factor
      typedef junction_tree<variable_type*, factor_type> jt_type;
      typedef typename jt_type::vertex vertex;
      jt_type jt(cliques(), factors().begin());

      // std::cout << jt << std::endl;
      
      // Extract all the leafs
      std::stack<vertex> leafs; 
      foreach(vertex v, jt.vertices())
        if(jt.out_degree(v) == 1) leafs.push(v);

      // Iteratively prune the dangling leafs
      while(!leafs.empty()) {
        // Prune the leaf v from its neighbor u
        vertex u = leafs.top();
        leafs.pop();
        // Is the node still a leaf or has it become a singleton?
        if (jt.out_degree(u) == 1) { 
          vertex v = *jt.neighbors(u).first;
          // Check if the leaf can be pruned
          if (includes(jt.clique(v), set_intersect(jt.clique(u), retain))) {
            jt[v].transfer_from(jt[u]);
            jt.remove_vertex(u);
            if (jt.out_degree(v) == 1) leafs.push(v);
          }
        }
      }

      // Now form a decomposable fragment using the remaining PL factors
      return decomposable_fragment(jt.vertex_properties(),
                                   set_intersect(arguments(), retain));
    }

    //! implements Factor::restrict_
    //! Note: decomposable_fragments do not support the restrict operation
    decomposable_fragment restrict(const assignment_type& a) const {
      throw std::invalid_argument("Unsupported operation");
    }

    //! implements Factor::subst_args
    decomposable_fragment& 
    subst_args(const std::map<variable_type*, variable_type*>& map) {
      args = subst_vars(args, map);
      foreach(factor_type& pl, *pls) pl.subst_args(map);
      return *this;
    }

    //! implements DistributionFactor::marginal
    decomposable_fragment marginal(const domain_type& retain) const {
      return collapse(retain, sum_op);
    }

  }; // class decomposable_fragment

  //! Combines two decomposable fragments
  //! \relates decomposable_fragment
  template <typename F, typename G>
  decomposable_fragment<F,G>
  combine(const decomposable_fragment<F,G>& x, 
          const decomposable_fragment<F,G>& y, op_type op) {
    factor::check_supported(op, product_op);
    return decomposable_fragment<F,G>(make_joined(x.factors(), y.factors()),
				      x.arguments() + y.arguments());
  }

  //! Combines a decomposable fragment and a likelihood
  //! \relates decomposable_fragment
  template <typename F, typename G>
  decomposable_fragment<F,G>
  combine(decomposable_fragment<F,G> x, const G& likelihood, op_type op) {
    return x.combine_in(likelihood, op);
  }

  //! Combines a decomposable fragment and a likelihood
  //! \relates decomposable_fragment
  template <typename F, typename G>
  decomposable_fragment<F,G>
  combine(const G& likelihood, decomposable_fragment<F,G> x, op_type op) {
    return x.combine_in(likelihood, op);
  }

  template <typename F, typename G>
  struct combine_result<decomposable_fragment<F,G>, decomposable_fragment<F,G> >
  {
    typedef decomposable_fragment<F,G> type;
  };

  template <typename F, typename G>
  struct combine_result<decomposable_fragment<F,G>, G > {
    typedef decomposable_fragment<F,G> type;
  };

  template <typename F, typename G>
  struct combine_result<G, decomposable_fragment<F,G> > {
    typedef decomposable_fragment<F,G> type;
  };

  //! Prints a model fragment to a stream
  //! relates decomposable_fragment
  template <typename F, typename G>
  std::ostream& 
  operator<<(std::ostream& out, const decomposable_fragment<F,G>& df) {
    out << "#F(DF|" << df.arguments() << "|" << df.cliques() << ")";
    return out;
  }

} // namespace prl

#include <prl/macros_undef.hpp>

#endif




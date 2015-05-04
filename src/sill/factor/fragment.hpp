#ifndef SILL_FRAGMENT_HPP
#define SILL_FRAGMENT_HPP

#include <stack>
#include <map>

#include <sill/global.hpp>
#include <sill/factor/base/factor.hpp>
#include <sill/factor/prior_likelihood.hpp>
#include <sill/factor/traits.hpp>
#include <sill/model/decomposable.hpp>
#include <sill/range/joined.hpp>

#include <sill/macros_def.hpp>

namespace sill {
  
  /**
   * Decomposable fragment is a factor that represents a subset 
   * of marginals of the probability distribution. Under certain
   * conditions, the edges of the true model can be identified 
   * locally from the cliques of the factor, which can be used
   * to perform marginalization operations.
   *
   * Models: Factor
   *
   * \tparam F the underlying factor type. Must satisfy the
   *           DistributionFactor concept.
   *
   * \ingroup factor_types
   * \see Factor
   */
  template <typename F>
  class fragment : public factor {
    concept_assert((DistributionFactor<F>));
    
    // Public type declarations
    //==========================================================================
  public:
    // Factor concept types
    typedef typename F::result_type     result_type;
    typedef typename F::real_type       real_type;
    typedef typename F::variable_type   variable_type;
    typedef typename F::domain_type     domain_type;
    typedef typename F::var_vector_type var_vector_type;
    typedef typename F::assignment_type assignment_type;

    // Other types
    typedef prior_likelihood<F> factor_type;

    // Private data members
    //==========================================================================
  private:
    //! The type that stores factors in this fragment
    typedef std::vector<prior_likelihood<F> > pl_vector;
    
    //! The argument set of this factor
    domain_type args;

    //! The collection of factors that forms this fragment
    //! \todo deprecate the copy_ptr
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
    fragment() : pls(new pl_vector()) { }

    //! Singleton constructor
    explicit fragment(const prior_likelihood<F>& pl) 
      : args(pl.arguments()),
        pls(new pl_vector(1, pl)) { }

    //! Converts a prior to a decomposable fragment. 
    //! Care must be taken that fragment is not
    //! constructed with a likelihood alone.
    explicit fragment(const F& prior)
      : args(prior.arguments()),
        pls(new pl_vector(1, factor_type(prior))) { }

    //! Conversion constructor
    explicit fragment(double val)
      : pls(new pl_vector(1, factor_type(val))) { }

    //! Constructs a decomposable fragment for a collection of PL factors 
    //! The argument set will be the union of all of factors' argument sets
    template <typename Range>
    explicit fragment(const Range& factors) {
      concept_assert((InputRangeConvertible<Range, factor_type>));
      pls.reset(new pl_vector(boost::begin(factors), boost::end(factors)));
      args = sill::arguments(*pls); // from factor/opereations.hpp
    }

    //! Construct a decomposable fragment for a collection of PL factors
    template <typename Range>
    explicit fragment(const Range& factors, const domain_type& args)
      : args(args) {
      concept_assert((InputRangeConvertible<Range, factor_type>));
      pls.reset(new pl_vector(boost::begin(factors), boost::end(factors)));
    }

//     //! Conversion to a constant factor. \todo For now, always returns 1
//     operator constant_factor() const {
//       assert(arguments().empty());
//       return 1;
//     }

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

    //! Returns the collection of prior-likelihoods, contained in this fragment
    const pl_vector& factors() const {
      return *pls;
    }

    //! Returns the collection of priors, contained in this fragment
    forward_range<const F&> priors() const {
      return make_transformed(*pls, std::mem_fun_ref(&factor_type::prior));
    }
    
    //! Returns the collection of likelihoods, contained in this fragment
    forward_range<const F&> likelihoods() const {
      return make_transformed(*pls,std::mem_fun_ref(&factor_type::likelihood));
    }

    //! Returns the cliques, contained in this fragment
    forward_range<const domain_type&> cliques() const {
      return make_transformed(*pls, std::mem_fun_ref(&factor_type::arguments));
    }

    //! Returns true if two decomposable fragments are equal
    //! (i.e., have the same arguments and the same list of components).
    bool operator==(const fragment& other) const {
      assert(pls);
      assert(other.pls);
      return arguments() == other.arguments() && 
        (pls == other.pls || *pls == *other.pls);
    }

    //! Returns true if two decomposable fragments are not equal
    bool operator!=(const fragment& other) const {
      return !operator==(other);
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

    //! multiplies in another decomposable fragment
    fragment& operator*=(const fragment& x) {
      pls->insert(pls->begin(), x.pls->begin(), x.pls->end());
      args.insert(x.args.begin(), x.args.end());
      return *this;
    }

    //! multiplies in a likelihood
    //! the arguments of the likelihood must be covered by at least one clique
    fragment& operator*=(const F& likelihood) {
      foreach(factor_type& pl, *pls) {
        if (includes(pl.arguments(), likelihood.arguments())) {
          pl *= likelihood;
          return *this;
        }
      }
      throw std::invalid_argument("The arguments of the likelihood are not covered by any prior");
    }

    //! multiplies in a constant
    fragment& operator*=(result_type val) {
      assert(!pls->empty());
      pls->front() *= val;
      return *this;
    }

    //! computes a marginal over set of variables
    //! this is a lazy operations: as many cliques are pruned as possible
    fragment marginal(const domain_type& retain) const {
      if (includes(retain, arguments())) return *this; // not much to do
      
      // Construct the canonical tree
      // Each vertex of the tree is associated with a PL factor
      typedef junction_tree<variable_type, factor_type> jt_type;
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
      return fragment(jt.vertex_properties(),
                                   set_intersect(arguments(), retain));
    }

    //! implements Factor::restrict_
    //! Note: fragments do not support the restrict operation
    fragment restrict(const assignment_type& a) const {
      throw std::invalid_argument("Unsupported operation");
    }

    //! implements Factor::subst_args
    fragment& 
    subst_args(const std::map<variable_type, variable_type>& map) {
      args = subst_vars(args, map);
      foreach(factor_type& pl, *pls) pl.subst_args(map);
      return *this;
    }

  }; // class fragment

  //! Multiplies two decomposable fragments
  //! \relates fragment
  template <typename F>
  fragment<F>
  operator*(const fragment<F>& x,
            const fragment<F>& y) {
    return fragment<F>(make_joined(x.factors(), y.factors()),
				      set_union(x.arguments(), y.arguments()));
  }

  //! Multiplies a decomposable fragment and a likelihood
  //! \relates fragment
  template <typename F>
  fragment<F>
  operator*(fragment<F> x, const F& likelihood) {
    return x *= likelihood;
  }

  //! Multiplies a decomposable fragment and a likelihood
  //! \relates fragment
  template <typename F>
  fragment<F>
  operator*(const F& likelihood, fragment<F> x) {
    return x *= likelihood;
  }

  //! Prints a model fragment to a stream
  //! relates fragment
  template <typename F>
  std::ostream& 
  operator<<(std::ostream& out, const fragment<F>& df) {
    out << "#F(DF|" << df.arguments() << "|" << df.cliques() << ")";
    return out;
  }

  // Traits
  //============================================================================

  //! \addtogroup factor_traits
  //! @{

  template <typename F>
  struct has_multiplies<fragment<F> > : public boost::true_type { };

  template <typename F>
  struct has_multiplies_assign<fragment<F> > : public boost::true_type { };

  template <typename F>
  struct has_marginal<fragment<F> > : public boost::true_type { };
  
  //! @}

} // namespace sill

#include <sill/macros_undef.hpp>

#endif




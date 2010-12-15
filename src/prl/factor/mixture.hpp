#ifndef PRL_MIXTURE_HPP
#define PRL_MIXTURE_HPP

#include <vector>
#include <sstream>
#include <set>
#include <map>

#include <prl/factor/concepts.hpp>
#include <prl/factor/constant_factor.hpp>
#include <prl/factor/factor.hpp>
#include <prl/factor/operations.hpp>
#include <prl/global.hpp>
#include <prl/serialization/serialize.hpp>

#include <prl/macros_def.hpp>

namespace prl {
  
  /**
   * A class that represents a mixture (sum) of functions.
   * Conceptually, mixture is an array of factors of fixed size.
   *
   * We may change mixture to be a model (rather than factor).
   *
   * @tparam F the factor type of the mixture component.
   *        The factor type must support unnormalized functions.
   *
   * \ingroup factor_types
   * \see Factor
   */
  template <typename F>
  class mixture : public factor {
    concept_assert((DistributionFactor<F>));
    
    // Public type declarations
    //==========================================================================
  public:
    //! The number representation of the factor
    typedef typename F::result_type result_type;
    
    //! implements Factor::domain_type
    typedef typename F::domain_type domain_type;

    //! implements Factor::variable_type
    typedef typename F::variable_type variable_type;

    //! The result of a collapse operation
    typedef mixture collapse_type;

    //! implements Factor::collapse_ops
    static const unsigned collapse_ops = 1 << sum_op;

    //! implements Factor::combine_ops
    static const unsigned combine_ops = F::combine_ops;
    
    //! The base type
    typedef factor base;

    //! The type of assignments used in this factor
    typedef std::map<variable_type*, typename variable_type::value_type> 
      assignment_type;

    //! implements Factor::record_type
    typedef typename F::record_type record_type;

    // Private data members
    //==========================================================================
  private:
    //! The mixture components
    std::vector<F> comps;

    
    // Constructors and conversion operators
    //==========================================================================
  public:

    void serialize(oarchive & ar) const {
      ar << comps;
    }
    void deserialize(iarchive & ar){
      ar >> comps;
    }

    //! Initializes the factor to a fixed value
    mixture(double value = double()) : comps(1, F(value)) {}

    //! Initialize the factor to the given arguments and number of components
    mixture(size_t k, const domain_type& args) : comps(k, F(args)) {
      assert(k > 0);
    }

    //! Initialize the mixture to be the sum of k identical components
    mixture(size_t k, const F& init) : comps(k, init) {
      assert(k > 0);
    }

    //! Constructs a mixture with a single component
    mixture(const constant_factor& c) : comps(1, c) { }

    //! Constructs a mixture with a single component
    mixture(const F& x) : comps(1, x) { }

    //! Conversion to a single component
    operator F() const {
      assert(size() == 1);
      return comps[0];
    }

    //! Conversion to a constant (the mixture must have no arguments)
    operator constant_factor() const {
      assert(this->arguments().empty());
      double result = 0;
      for(size_t i = 0; i < comps.size(); i++) 
        result += constant_factor(comps[i]).value;
      return result;
    }

    //! Conversion to human-readable representation
    operator std::string() const {
      std::ostringstream out; out << *this; return out.str(); 
    }

    //! Swaps two factors
    void swap(mixture& f) {
      comps.swap(f.comps);
    }

    // Accessors and comparison operators
    //==========================================================================
    //! Returns the argument set of this factor
    const domain_type& arguments() const {
      return comps[0].arguments();
    }

    //! Returns the number of components of this mixture
    size_t size() const { return comps.size(); }

    //! Returns a component of the mixture
    const F& operator[](size_t i) const { return comps.at(i); }

    //! Mutable access to the component of the mixture
    //! The caller must not modify the argument set of the component.
    F& operator[](size_t i) { return comps.at(i); }

    //! Returns the components of the mixture
    const std::vector<F>& components() const { return comps; }

    //! Returns true if the two mixtures are the same
    bool operator==(const mixture& other) const {
      return arguments() == other.arguments() && comps == other.comps;
    }

    //! Returns true if the two mixtures are not the same
    bool operator!=(const mixture& other) const {
      return !operator==(other);
    }

    //! Evaluates the probability of an assignment
    double operator()(const assignment_type& a) const {
      double result = 1;
      foreach(const F& factor, comps) result += factor(a);
      return result;
    }
 
    // Factor operations
    //==========================================================================
    /*
    //! implements element-wise factor combination
    mixture& combine_in(const F& factor, op_type op) {
      args.insert(factor.arguments());
      foreach(F& comp, comps) comp.combine_in(factor, op);
      return *this;
    }
    */

    //! implements Factor::combine_in
    mixture& combine_in(const constant_factor& other, op_type op) {
      for(size_t i = 0; i < comps.size(); i++)
        comps[i].combine_in(other, op);
      return *this;
    }

    //! implements Factor::combine_in
    //! \todo For now, the combination is element-wise; this will change
    mixture& combine_in(const mixture& other, op_type op) {
      assert(size() == other.size());
      for(size_t i = 0; i < other.size(); i++)
        comps[i].combine_in(other[i], op);
      return *this;
    }

    //! implements Factor::collapse
    mixture collapse(const domain_type& retained, op_type op) const {
      check_supported(op, collapse_ops);
      
      // If the retained arguments contain the arguments of this factor,
      // we can simply return a copy
      if (arguments().subset_of(retained)) return *this;

      // Collapse the individual components
      domain_type newargs = arguments().intersect(retained);
      mixture factor(size(), newargs);

      for(size_t i = 0; i < size(); i++)
        factor.comps[i] = comps[i].collapse(retained, sum_op);
      
      return factor;
    }

    //! implements Factor::restrict
    mixture restrict(const assignment_type& a) const {
      domain_type bound_vars = a.keys();
      
      // If the arguments are disjoint from the bound variables,
      // we can simply return a copy of the factor
      if (!bound_vars.meets(arguments())) return *this;

      // Restrict each component factor
      domain_type retained = arguments() - bound_vars;
      mixture factor(size(), retained);
      
      for(size_t i = 0; i < size(); i++)
        factor.comps[i] = comps[i].restrict(a);
    
      return factor;
    }

    //! Adds the parameters of another mixture
    mixture& add_parameters(const mixture& other, double w = 1) {
      assert(size() == other.size());
      for(size_t i = 0; i < size(); i++)
        comps[i].add_parameters(other[i], w);
      return *this;
    }

    //! implements Factor::subst_args
    mixture& subst_args(const std::map<variable_type*, variable_type*>& var_map) {
      foreach(F& factor, comps) factor.subst_args(var_map);
      return *this;
    }

    //! implements DistributionFactor::marginal
    mixture marginal(const domain_type& retain) const {
      return collapse(retain, sum_op);
    }

    //! implements DistributionFactor::is_normalizable
    bool is_normalizable() const {
      foreach(const F& factor, comps)
        if (factor.is_normalizable()) return true;
      return false;
    }

    //! implements DistributionFactor::norm_constant
    //! \todo this operation may not be num. stable 
    //! in some cases, we should be using the log-domain
    double norm_constant() const {
      double p = 0;
      foreach(const F& factor, comps) p += factor.norm_constant();
      return p;
    }

    //! implements DistributionFactor::normalize
    //! uses the standard parameterization
    mixture& normalize() {
      double p = norm_constant();
      assert(p>0);
      foreach(F& factor, comps) factor /= constant_factor(p);
      return *this;
    }

    //! implements DistributionFactor::entropy
    double entropy() const {
      assert(false); return 0; // not supported yet - hard to do exactly
    }

    //! implements DistributionFactor::relative_entropy
    double relative_entropy(const mixture& other) const {
      assert(false); return 0; // not supported yet - hard to do exactly
    }

    //! this will be moved elsewhere
    assignment_type arg_max() const {
      assert(false); return assignment_type(); // not supported
    }

    //! this will be moved elsewhere
    assignment_type arg_min() const {
      assert(false); return assignment_type(); // not supported
    }

  };

  //! \relates mixture
  template <typename F>
  std::ostream& operator<<(std::ostream& out, const mixture<F>& mixture) {
    out << "#F(M | " << mixture.arguments();
    foreach(const F& factor, mixture.components())
      out << " | " << factor;
    out << ")";
    return out;
  }

  // Mixture combinations
  //============================================================================
  //! \relates mixture
  template <typename F>
  mixture<F> combine(mixture<F> x, const mixture<F>& y, op_type op) {
    return x.combine_in(y, op);
  }

  //! \relates mixture
  template <typename F>
  mixture<F> combine(mixture<F> x, const F& y, op_type op) {
    return x.combine_in(y, op);
  }

  //! \relates mixture
  template <typename F>
  mixture<F> combine(const F& x, mixture<F> y, op_type op) {
    return y.combine_in(x, op);
  }
  
  template <typename F>
  struct combine_result< mixture<F>, mixture<F> > {
    typedef mixture<F> type;
  };

  template <typename F>
  struct combine_result< mixture<F>, F > {
    typedef mixture<F> type;
  };

  template <typename F>
  struct combine_result< F, mixture<F> > {
    typedef mixture<F> type;
  };

  // Free functions
  //============================================================================
  class moment_gaussian;

  //! A mixture-Gaussian distribution
  //! \relates mixture
  //! \ingroup factor_types
  typedef mixture<moment_gaussian> mixture_gaussian;

  //! Computes the KL projection of a mixture of Gaussians to a Gausian
  //! using moment matching.
  //! \relates mixture
  moment_gaussian project(const mixture_gaussian& mixture);

} // namespace prl

#endif // PRL_MIXTURE_HPP

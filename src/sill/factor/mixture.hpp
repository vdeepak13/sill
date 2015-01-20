#ifndef SILL_MIXTURE_HPP
#define SILL_MIXTURE_HPP

#include <vector>
#include <sstream>
#include <set>
#include <map>

#include <sill/global.hpp>
#include <sill/factor/concepts.hpp>
#include <sill/factor/factor.hpp>
#include <sill/factor/moment_gaussian.hpp>
#include <sill/factor/util/operations.hpp>
#include <sill/factor/traits.hpp>
#include <sill/functional/assign.hpp>
#include <sill/serialization/serialize.hpp>

#include <sill/macros_def.hpp>

namespace sill {
  
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
    
    //! implements Factor::variable_type
    typedef typename F::variable_type variable_type;

    //! implements Factor::domain_type
    typedef typename F::domain_type domain_type;

    //! The type of assignments used in this factor
    typedef typename F::assignment_type assignment_type;

    //! implements Factor::record_type
    typedef typename F::record_type record_type;

    typedef typename F::var_vector_type var_vector_type;


    // Constructors, conversion operators, and serialization
    //==========================================================================
  public:

    //! Initializes the factor to a fixed value
    explicit mixture(double value = 1) : comps(1, F(value)) {}

    //! Initialize the factor to the given arguments and number of components
    mixture(size_t k, const domain_type& args) : comps(k, F(args)) {
      assert(k > 0);
    }

    //! Initializes the factor to teh given arguments and number of components
    mixture(size_t k, const var_vector_type& args) : comps(k, F(args)) {
      assert(k > 0);
    }

    //! Initialize the mixture to be the sum of k identical components
    mixture(size_t k, const F& init) : comps(k, init) {
      assert(k > 0);
    }

    //! Constructs a mixture with a single component
    explicit mixture(const F& x) : comps(1, x) { }

    //! Conversion to a single component
    operator F() const {
      assert(size() == 1);
      return comps[0];
    }

//     //! Conversion to a constant (the mixture must have no arguments)
//     operator constant_factor() const {
//       assert(this->arguments().empty());
//       double result = 0;
//       for(size_t i = 0; i < comps.size(); i++) 
//         result += constant_factor(comps[i]).value;
//       return result;
//     }

    //! Conversion to human-readable representation
    operator std::string() const {
      std::ostringstream out; out << *this; return out.str(); 
    }

    //! Swaps two factors
    void swap(mixture& f) {
      comps.swap(f.comps);
    }

    void save(oarchive & ar) const {
      ar << comps;
    }

    void load(iarchive & ar){
      ar >> comps;
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

    //! Returns the components of the mixture
    std::vector<F>& components() { return comps; }

    //! Returns true if the two mixtures are the same
    bool operator==(const mixture& other) const {
      return arguments() == other.arguments() && comps == other.comps;
    }

    //! Returns true if the two mixtures are not the same
    bool operator!=(const mixture& other) const {
      return !operator==(other);
    }

    // Factor operations
    //==========================================================================

    //! Evaluates the probability of an assignment
    double operator()(const assignment_type& a) const {
      if (comps.size() == 0)
        return 1;
      double result = 0;
      foreach(const F& factor, comps) result += factor(a);
      return result;
    }

    //! Returns the value associated with an assignment.
    double logv(const vector_assignment& a) const {
      return std::log(operator()(a));
    }

    //! multiplies each component by a constant
    mixture& operator*=(result_type val) {
      foreach(F& factor, comps) { factor *= val; }
      return *this;
    }

    //! divides each component by a constant
    mixture& operator/=(result_type val) {
      foreach(F& factor, comps) { factor /= val; }
      return *this;
    }

    //! multiplies each component by a factor (defined if F supports multiplication)
//     template <typename G=F,
//               typename boost::enable_if<has_multiplies_assign<F> >::type* = NULL>
    mixture& operator*=(const F& factor) {
      return componentwise_op(factor, multiplies_assign<F>());
    }

    //! divides each component by a factor (defined if F supports division)
//     template <typename G=F,
//               typename boost::enable_if<has_divides_assign<F> >::type* = NULL>
    mixture& operator/=(const F& factor) {
      return componentwise_op(factor, divides_assign<F>());
    }

    //! component-wise multiplication (defined if F supports multiplication)
//     template <typename G=F,
//               typename boost::enable_if<has_multiplies_assign<F> >::type* = NULL>
    mixture& operator*=(const mixture& other) {
      return componentwise_op(other, multiplies_assign<F>());
    }

    //! component-wise division (defined if F supports division)
//     template <typename G=F,
//               typename boost::enable_if<has_divides_assign<F> >::type* = NULL>
    mixture& operator/=(const mixture& other) {
      return componentwise_op(other, divides_assign<F>());
    }

    //! Computes a marginal of the mixture over a subset of variables
    mixture marginal(const domain_type& retained) const {
      // If the retained arguments contain the arguments of this factor,
      // we can simply return a copy
      if (includes(retained, arguments()))
        return *this;

      // Collapse the individual components
      mixture m(size(), set_intersect(retained, arguments()));
      for(size_t i = 0; i < size(); i++)
        m.comps[i] = comps[i].marginal(retained);

      return m;
    }

    //! Computes a marginal, storing the result in m.
    //! This version does not update the normalization constant.
    //! Todo: actually avoid reallocation if possible
    void marginal_unnormalized(const domain_type& retain, mixture& m) const {
      // Collapse the individual components
      m = mixture(size());
      for(size_t i = 0; i < size(); i++)
        comps[i].marginal_unnormalized(retain, m.comps[i]);
    }

    //! implements Factor::restrict
    mixture restrict(const assignment_type& a) const {
      domain_type bound_vars = keys(a);

      // If the arguments are disjoint from the bound variables,
      // we can simply return a copy of the factor
      if (set_disjoint(bound_vars, arguments()))
        return *this;

      // Restrict each component factor
      domain_type retained = set_difference(arguments(), bound_vars);
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
    mixture&
    subst_args(const std::map<variable_type*, variable_type*>& var_map) {
      foreach(F& factor, comps) factor.subst_args(var_map);
      return *this;
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
      if (comps.size() > 0) {
        double p = norm_constant();
        assert(p > 0);
        foreach(F& factor, comps)
          factor /= p;
      }
      return *this;
    }

    //! Sample from the mixture.
    template <typename Engine>
    assignment_type sample(Engine& rng) const {
      if (comps.size() == 0)
        return assignment_type();
      boost::uniform_int<int> unif_int(0, comps.size() - 1);
      size_t i = unif_int(rng);
      return comps[i].sample(rng);
    }

    // Private data members
    //==========================================================================
  private:

    //! The mixture components
    std::vector<F> comps;

    template <typename Op>
    mixture& componentwise_op(const F& factor, Op op) {
      foreach(F& comp, comps) {
        op(comp, factor);
      }
      return *this;
    }

    template <typename Op>
    mixture& componentwise_op(const mixture& other, Op op) {
      if (size() != other.size()) {
        throw std::runtime_error
          ("mixture::combine_in(other,op) given other with mismatched size.");
      }
      for(size_t i = 0; i < other.size(); i++) {
        op(comps[i], other[i]);
      }
      return *this;
    }

  }; // class mixture

  //! \relates mixture
  template <typename F>
  std::ostream& operator<<(std::ostream& out, const mixture<F>& mixture) {
    out << "#F(M | " << mixture.arguments();
    foreach(const F& factor, mixture.components())
      out << "\n | " << factor;
    out << ")\n";
    return out;
  }

  // Mixture combinations
  //============================================================================

  //! Multiplies the mixture by a constant
  //! \relates mixture
  template <typename F>
  mixture<F> operator*(mixture<F> x, typename F::result_type val) {
    return x *= val;
  }

  //! Multiplies the mixture by a constant
  //! \relates mixture
  template <typename F>
  mixture<F> operator*(typename F::result_type val, mixture<F> x) {
    return x *= val;
  }

  //! Divides the mixture by a constant
  //! \relates mixture
  template <typename F>
  mixture<F> operator/(mixture<F> x, typename F::result_type val) {
    return x /= val;
  }

  //! Multiplies the mixture by a factor (defined if F supports multiplication)
  //! \relates mixture
  template <typename F>
  mixture<F> operator*(mixture<F> x, const F& factor) {
    return x *= factor;
  }

  //! Multiplies the mixture by a factor (defined if F supports multiplication)
  //! \relates mixture
  template <typename F>
  mixture<F> operator*(const F& factor, mixture<F> x) {
    return x *= factor;
  }

  //! Divides the mixture by a factor (defined if F supports division)
  //! \relates mixture
  template <typename F>
  mixture<F> operator/(mixture<F> x, const F& factor) {
    return x /= factor;
  }
  
  //! Multiplies two mixtures component-wise (defined if F supports multiplication)
  //! \relates mixture
  template <typename F>
  mixture<F> operator*(mixture<F> x, const mixture<F>& y) {
    return x *= y;
  }

  //! Divides two mixtures component-wise (defined if F supports division)
  //! \relates mixture
  template <typename F>
  mixture<F> operator/(mixture<F> x, const mixture<F>& y) {
    return x /= y;
  }
    
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

  // Traits
  //============================================================================

  //! \addtogroup factor_traits
  //! @{

  template <typename F>
  struct has_multiplies<mixture<F> > : public has_multiplies_assign<F> { };

  template <typename F>
  struct has_multiplies_assign<mixture<F> > : public has_multiplies_assign<F> { };

  template <typename F>
  struct has_divides<mixture<F> > : public has_divides_assign<F> { };

  template <typename F>
  struct has_divides_assign<mixture<F> > : public has_divides_assign<F> { };

  template <typename F>
  struct has_marginal<mixture<F > > : public boost::true_type { };

  //! @}

} // namespace sill

#endif // SILL_MIXTURE_HPP
